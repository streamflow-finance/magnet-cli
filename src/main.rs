use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::PathBuf,
    str::FromStr,
    time::Duration,
};

use clap::{Parser, Subcommand};
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use retry::{delay::Fixed, OperationResult};
use serde::{Deserialize, Serialize};
use solana_client::rpc_client::RpcClient;
use solana_program::pubkey::Pubkey;

use anyhow::{Context, Result};
use solana_sdk::account::Account;
use solana_transaction_status::{
    parse_accounts::ParsedAccount, parse_instruction::ParsedInstruction,
    EncodedConfirmedTransactionWithStatusMeta, UiInnerInstructions,
};

const STD_RETRY: u64 = 50;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Turn debugging information on
    #[clap(short, long, parse(from_occurrences))]
    debug: usize,

    /// Solana RPC url
    #[clap(short, long)]
    rpc_url: Option<String>,

    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// List program addresses
    ListPrograms {
        /// Sets an output file
        #[clap(short, long, parse(from_os_str))]
        output: PathBuf,

        /// Should program addresses that are not executable be printed
        #[clap(short, long)]
        print_non_executable: bool,
    },

    Analyze {
        /// Sets an input file
        #[clap(short, long, parse(from_os_str))]
        input: PathBuf,

        /// Sets an output file
        #[clap(short, long, parse(from_os_str))]
        output: PathBuf,

        /// Referent program address
        #[clap(short, long)]
        referent_addr: String,

        /// Transactions per program to analyze
        #[clap(short, long, default_value_t = 50)]
        tx_cnt: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let rpc_url = match cli.rpc_url {
        Some(url) => url.clone(),
        None => "https://api.mainnet-beta.solana.com".to_owned(),
    };

    let rpc_client = RpcClient::new_with_timeout(rpc_url, Duration::from_secs(90));

    let env = env_logger::Env::default()
        .filter_or(env_logger::DEFAULT_FILTER_ENV, "info")
        .write_style_or(env_logger::DEFAULT_WRITE_STYLE_ENV, "always");
    env_logger::init_from_env(env);

    match cli.command {
        Some(Commands::ListPrograms {
            output,
            print_non_executable,
        }) => {
            list_programs(rpc_client, output, print_non_executable)?;
        }
        Some(Commands::Analyze {
            input,
            output,
            referent_addr,
            tx_cnt,
        }) => {
            analyze(rpc_client, input, output, referent_addr, tx_cnt)?;
        }
        None => {
            panic!("Command must be specified");
        }
    }

    Ok(())
}

fn analyze_one(rpc: &RpcClient, addr: String, tx_cnt: usize) -> Result<ProgramInfo> {
    log::info!("Fetching info for program on addr={}", addr);
    let key = Pubkey::from_str(&addr)?;
    let program_acc = retry_n(STD_RETRY, || rpc.get_account(&key))?;

    if !program_acc.executable {
        return Err(anyhow::anyhow!("Program not executable!"));
    }

    let mut last_deployed_timestamp = 0 as i64;
    let mut program_size = 0;
    use solana_program::bpf_loader_upgradeable::UpgradeableLoaderState;
    let bpf_loader_state: UpgradeableLoaderState = program_acc.deserialize_data()?;
    if let UpgradeableLoaderState::Program {
        programdata_address,
    } = bpf_loader_state
    {

        let programdata_acc = retry_n(STD_RETRY, || rpc.get_account(&programdata_address))?;
        program_size = programdata_acc.data.len();

        use solana_account_decoder::parse_bpf_loader::{
            parse_bpf_upgradeable_loader, BpfUpgradeableLoaderAccountType,
        };
        if let Ok(BpfUpgradeableLoaderAccountType::ProgramData(data)) =
            parse_bpf_upgradeable_loader(&programdata_acc.data)
        {
            let deploy_block = retry_n(STD_RETRY, || rpc.get_block(data.slot))?;
            last_deployed_timestamp = deploy_block.block_time.unwrap_or(0);
            if last_deployed_timestamp != 0 {
                log::info!(" program last deployed at TS: {}", last_deployed_timestamp);
            }
        }
            
        log::info!(
            "  Program data size extracted = {} from addr {}",
            program_size,
            programdata_address
        );
    }

    let mut size_occurencies = HashMap::new();

    let program_accs = retry_n(STD_RETRY, || rpc.get_program_accounts(&key))?;
    for (_, acc) in program_accs {
        let byte_size = acc.data.len();
        *size_occurencies.entry(byte_size).or_insert(0usize) += 1;
    }

    let sigs = retry_n(STD_RETRY, || {
        let cfg = solana_client::rpc_client::GetConfirmedSignaturesForAddress2Config {
            limit: Some(tx_cnt),
            ..Default::default()
        };
        rpc.get_signatures_for_address_with_config(&key, cfg)
    })?;

    log::info!(
        "Extracting tx info for {} transactions for addr {}",
        tx_cnt,
        addr
    );

    let tx_infos: Vec<_> = sigs
        .par_iter()
        .filter_map(|signature| {
            let sign = solana_sdk::signature::Signature::from_str(&signature.signature).unwrap();
            let tx = retry::retry_with_index(Fixed::from_millis(1_000), |ix| {
                if ix >= 20 {
                    return OperationResult::Err("did not succeed within 20 tries");
                }
                let tx = rpc.get_transaction(
                    &sign,
                    solana_transaction_status::UiTransactionEncoding::JsonParsed,
                );

                if tx.is_err() {
                    let err = format!("{:?}", tx.unwrap_err());
                    if (err.contains("Connection refused") || err.contains("os error 61"))
                        || err.contains("connection error")
                        || err.contains("hyper::Error")
                        || err.contains("502 Bad Gateway")
                    {
                        return OperationResult::Retry("connection error - should retry");
                    } else {
                        return OperationResult::Err("other error happened, retry!");
                    }
                } else {
                    return OperationResult::Ok(tx.unwrap());
                }
            });

            if tx.is_err() {
                return None;
            }
            extract_info_from_tx(&signature.signature, tx.unwrap())
        })
        .collect();

    let set_of_parsed_ins = set_of_parsed_inner_instructions(&tx_infos);

    Ok(ProgramInfo {
        addr,
        program_size,
        last_deployed_ts: last_deployed_timestamp,
        size_occurencies,
        tx_infos,
        set_of_parsed_ins,
    })
}

fn retry_n<F, T>(max: u64, mut func: F) -> Result<T>
where
    F: FnMut() -> std::result::Result<T, solana_client::client_error::ClientError>,
    T: core::fmt::Debug,
{
    let res = retry::retry_with_index(Fixed::from_millis(1_100), |ix| {
        if ix >= max {
            return OperationResult::Err("did not succeed within MAX tries");
        }
        let res = func();

        if res.is_err() {
            let err = format!("{:?}", res.unwrap_err());
            if err.contains("Connection refused")
                || err.contains("os error 61")
                || err.contains("connection error")
                || err.contains("hyper::Error")
                || err.contains("502 Bad Gateway")
            {
                return OperationResult::Retry("connection error - should retry");
            } else {
                log::warn!("Unrecoverable error happened, will drop - {:?}", err);
                return OperationResult::Err("Non retryable error occured!");
            }
        } else {
            return OperationResult::Ok(res.unwrap());
        }
    });

    match res {
        Ok(r) => return Ok(r),
        Err(e) => return Err(anyhow::anyhow!("Error while executing - {:?}", e)),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProgramInfo {
    addr: String,
    program_size: usize,
    last_deployed_ts: i64,
    size_occurencies: HashMap<usize, usize>,
    tx_infos: Vec<TxInfo>,
    set_of_parsed_ins: HashSet<String>,
}

impl ProgramInfo {
    fn tx_infos_by_shape(&self) -> HashMap<Vec<(bool, bool)>, Vec<TxInfo>> {
        let tx_infos = self.tx_infos.clone();

        let mut tx_info_m = HashMap::new();
        for tx_info in tx_infos {
            let acc_shape = tx_info.acc_shape();
            (*tx_info_m.entry(acc_shape).or_insert(vec![])).push(tx_info);
        }
        tx_info_m
    }

    fn program_owned_accounts_cnt(&self) -> usize {
        self.size_occurencies.values().sum()
    }

    fn program_owned_accounts_avg_size(&self) -> usize {
        let sum_cnt = self.program_owned_accounts_cnt();
        self.size_occurencies
            .iter()
            .map(|(size, cnt)| size * cnt / sum_cnt)
            .sum()
    }
}

fn extract_info_from_tx(
    signature: &str,
    tx: EncodedConfirmedTransactionWithStatusMeta,
) -> Option<TxInfo> {
    let tx_meta = tx.transaction.meta;
    if let None = tx_meta {
        return None;
    }

    let tx_meta = tx_meta.unwrap();
    if let solana_transaction_status::EncodedTransaction::Json(ui_ix) = tx.transaction.transaction {
        if let solana_transaction_status::UiMessage::Parsed(msg) = ui_ix.message {
            let inner_inst = tx_meta.inner_instructions;
            let mut inner_parsed_instructions = vec![];

            if let Some(inner_insts) = &inner_inst {
                for inst in inner_insts {
                    for iii in &inst.instructions {
                        use solana_transaction_status::{UiInstruction, UiParsedInstruction};
                        match iii {
                            UiInstruction::Parsed(UiParsedInstruction::Parsed(p)) => {
                                inner_parsed_instructions.push(p.clone())
                            }
                            _ => continue,
                        }
                    }
                }
            }

            return Some(TxInfo {
                signature: signature.to_owned(),
                logs: tx_meta.log_messages,
                acc_keys: msg.account_keys,
                inner_intructions: inner_inst,
                inner_parsed_instructions,
            });
        }
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TxInfo {
    signature: String,
    logs: Option<Vec<String>>,
    acc_keys: Vec<ParsedAccount>,
    inner_intructions: Option<Vec<UiInnerInstructions>>,
    inner_parsed_instructions: Vec<ParsedInstruction>,
}

impl TxInfo {
    fn acc_shape(&self) -> Vec<(bool, bool)> {
        self.acc_keys
            .iter()
            .map(|a| (a.signer, a.writable))
            .collect()
    }
}

fn list_programs(rpc: RpcClient, output: PathBuf, print_non_executable: bool) -> Result<()> {
    let bpf_loader_upgradeable = Pubkey::from_str("BPFLoaderUpgradeab1e11111111111111111111111")?;
    let bpf_loader = Pubkey::from_str("BPFLoader2111111111111111111111111111111111")?;
    let bpf_loader_deprecated = Pubkey::from_str("BPFLoader1111111111111111111111111111111111")?;

    log::info!("Fetching program accounts...");

    let mut res = fetch_program_accounts(&rpc, &bpf_loader_upgradeable)?;
    res.extend(fetch_program_accounts(&rpc, &bpf_loader)?);
    res.extend(fetch_program_accounts(&rpc, &bpf_loader_deprecated)?);

    let mut out = std::fs::File::create(&output).with_context(|| {
        format!(
            "Unable to create output file on path `{}`",
            output.to_string_lossy()
        )
    })?;

    let all = res.len();
    let mut non_executable = 0;
    let mut executable = 0;

    for (addr, account) in res {
        if !account.executable {
            non_executable += 1;
            if print_non_executable {
                writeln!(&mut out, "{}", addr.to_string())?;
            }
        } else {
            executable += 1;
            writeln!(&mut out, "{}", addr.to_string())?;
        }
    }

    log::info!(
        "In sum there are {} programs, {} are executable, and {} are not.",
        all,
        executable,
        non_executable
    );

    out.flush().context("Error while flushing output!")?;
    Ok(())
}

fn fetch_program_accounts(rpc: &RpcClient, key: &Pubkey) -> Result<Vec<(Pubkey, Account)>> {
    let res = rpc
        .get_program_accounts(&key)
        .with_context(|| format!("Error while fetching program accounts owhned by {}", key))?;

    log::info!("There are {} programs listed owned by `{}`", res.len(), key);
    Ok(res)
}

fn analyze(
    rpc: RpcClient,
    input: PathBuf,
    output: PathBuf,
    referent_addr: String,
    tx_cnt: usize,
) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(16, tx_cnt / 5))
        .build_global()
        .unwrap();

    let mut out = std::fs::File::create(&output).with_context(|| {
        format!(
            "Unable to create output file on path `{}`",
            output.to_string_lossy()
        )
    })?;

    let referent_info = analyze_one(&rpc, referent_addr, std::cmp::min(4 * tx_cnt, 1000))?;

    let referent_tx_infos_by_shape = referent_info.tx_infos_by_shape();

    log::info!(
        "referent info = program_size:{}, size_occs: {:?}, txs_cnt: {}, different shapes cnt: {}",
        referent_info.program_size,
        referent_info.size_occurencies,
        referent_info.tx_infos.len(),
        referent_tx_infos_by_shape.keys().count(),
    );
    log::info!(
        "referent info - instructions parsed set {:?}",
        referent_info.set_of_parsed_ins
    );

    let tx_infos = referent_info.tx_infos.clone();

    let mut tx_info_m = HashMap::new();
    for tx_info in tx_infos {
        let acc_shape = tx_info.acc_shape();
        (*tx_info_m.entry(acc_shape).or_insert(vec![])).push(tx_info);
    }

    let mut keys = tx_info_m.keys();
    let one_key = keys.next().unwrap();
    let other_key = keys.next().unwrap();

    let txs1 = tx_info_m.get(one_key).unwrap();
    let txs2 = tx_info_m.get(other_key).unwrap();

    let cmp1 = cmp_tx_logs(&txs1[0], &txs1[1]);
    log::info!("CMP1 = {:?}", cmp1);

    let cmp2 = cmp_tx_logs(&txs1[0], &txs2[0]);
    log::info!("CMP2 = {:?}", cmp2);

    for addr in std::fs::read_to_string(input)?.lines() {
        let mut err_cnt = 0;
        let other_info = loop {
            let info = analyze_one(&rpc, addr.to_owned(), tx_cnt);

            if info.is_err() {
                err_cnt += 1;
                let err = format!("{:?}", info);
                if (err.contains("Connection refused") && err.contains("os error 61"))
                    || err.contains("connection error")
                    || err.contains("did not succeed within")
                {
                    if err_cnt % 10 == 0 {
                        log::warn!("Errored 10 times in row, dropping");
                        break info;
                    }
                    std::thread::sleep(Duration::from_secs(3));
                    continue;
                }
            }

            break info;
        };

        if other_info.is_err() {
            log::error!(
                "Dropping account address `{}` due to error: {:?}",
                addr,
                other_info
            );
            continue;
        }
        let other_info = other_info.unwrap();

        if other_info.size_occurencies.len() == 0 {
            log::info!(
                "Dropping account address `{}` since it has no associated program accounts",
                addr
            );
            continue;
        }

        if other_info.tx_infos.len() == 0 {
            log::info!(
                "Dropping account address `{}` since it has no confirmed transactions",
                addr
            );
            continue;
        }

        log::info!("Comparing referent with {}", other_info.addr);
        let cmp = cmp_two_programs(&referent_info, &other_info);

        log::info!("Compare finished - {:?}", cmp);
        writeln!(&mut out, "{}", serde_json::to_string(&cmp)?)?;
    }

    out.flush().context("Error while flushing output!")?;

    Ok(())
}

fn cmp_two_programs(p1: &ProgramInfo, p2: &ProgramInfo) -> ProgramCMP {
    let p1_tx_map = p1.tx_infos_by_shape();
    let p2_tx_map = p2.tx_infos_by_shape();

    let mut shape_hits = 0usize;
    let mut mean_levenshtein_dist_per_shape = vec![];
    let mut variance_levenshtein_dist_per_shape = vec![];

    let mut mean_sorensen_dice_per_shape = vec![];
    let mut variance_sorensen_dice_per_shape = vec![];

    for p1_shape in p1_tx_map {
        if p2_tx_map.contains_key(&p1_shape.0) {
            shape_hits += 1;
            let tx1s = &p1_shape.1;
            let tx2s = &p2_tx_map[&p1_shape.0];

            // limiting the cartisian product to 3k since I don't want it to go overboard
            let tx_pairs = tx1s
                .iter()
                .cartesian_product(tx2s.iter())
                .take(3_000)
                .collect_vec();

            let cmp_res: Vec<(usize, f64, f64)> = tx_pairs
                .par_iter()
                .filter_map(|(tx1, tx2)| cmp_tx_logs(tx1, tx2))
                .collect();

            if cmp_res.len() == 0 {
                mean_levenshtein_dist_per_shape.push(0.0);
                variance_levenshtein_dist_per_shape.push(0.0);
                mean_sorensen_dice_per_shape.push(0.0);
                variance_sorensen_dice_per_shape.push(0.0);
            } else {
                let ls = cmp_res.iter().map(|(_, _, l)| *l).collect_vec();
                let ls_mean = statistical::mean(&ls);
                let ls_variance = statistical::population_variance(&ls, Some(ls_mean));

                mean_levenshtein_dist_per_shape.push(ls_mean);
                variance_levenshtein_dist_per_shape.push(ls_variance);

                let sds = cmp_res.iter().map(|(_, s, _)| *s).collect_vec();
                let sd_mean = statistical::mean(&sds);
                let sd_variance = statistical::population_variance(&sds, Some(sd_mean));

                mean_sorensen_dice_per_shape.push(sd_mean);
                variance_sorensen_dice_per_shape.push(sd_variance);
                log::info!(
                    "--> cmp res LEVEN = {} var {} | SORENSEN_DICE = {} var {}",
                    ls_mean,
                    ls_variance,
                    sd_mean,
                    sd_variance
                );
            }
        }
    }

    let overlapping_parsed_ins = p1
        .set_of_parsed_ins
        .intersection(&p2.set_of_parsed_ins)
        .map(|s| s.to_owned())
        .collect_vec();
    let overlapping_parsed_ins_factor_referent =
        overlapping_parsed_ins.len() as f64 / p1.set_of_parsed_ins.len() as f64;
    let overlapping_parsed_ins_factor_current =
        overlapping_parsed_ins.len() as f64 / p2.set_of_parsed_ins.len() as f64;

    let freq_p1 = log_words_frequency_map(&p1.tx_infos);
    let mut freq_p2 = log_words_frequency_map(&p2.tx_infos);
    freq_p2.entry(ALL_KEYWORD.to_string()).or_insert(0);

    let p1_words_count = freq_p1[ALL_KEYWORD] as f64;
    let p2_words_count = freq_p2[ALL_KEYWORD] as f64;

    let words_frequency_score = freq_p1
        .iter()
        .filter(|(word, freq)| !word.contains(ALL_KEYWORD) && **freq > 5)
        .map(|(word, freq)| {
            let freq2 = *freq_p2.get(word).unwrap_or(&0);
            if freq2 == 0 {
                return 0.0;
            }

            let freq2 = freq2 as f64;
            let freq1 = *freq as f64;

            // todo: is this ok?
            ((freq1 / p1_words_count) * (freq2 / p2_words_count)).sqrt()
        })
        .sum::<f64>();

    ProgramCMP {
        program_addr: p2.addr.clone(),
        program_instruction_size: p2.program_size,
        program_instruction_size_factor: p1.program_size as f64 / p2.program_size as f64,
        program_owned_accounts_cnt: p2.program_owned_accounts_cnt(),
        program_owned_accounts_avg_size: p2.program_owned_accounts_avg_size(),
        program_owned_accounts_avg_size_factor: p1.program_owned_accounts_avg_size() as f64
            / p2.program_owned_accounts_avg_size() as f64,
        program_owned_accounts_size_distribution: p2.size_occurencies.clone(),
        instruction_shapes: p2_tx_map.len(),
        shape_hits,
        mean_levenshtein_dist_per_shape,
        variance_levenshtein_dist_per_shape,
        mean_sorensen_dice_per_shape,
        variance_sorensen_dice_per_shape,
        overlapping_parsed_ins,
        overlapping_parsed_ins_factor_referent,
        overlapping_parsed_ins_factor_current,
        words_frequency_score,
    }
}

const ALL_KEYWORD: &'static str = "__ALL_KEYWORD__";

fn log_words_frequency_map(txs: &[TxInfo]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for tx in txs {
        let words = tx
            .logs
            .as_ref()
            .unwrap()
            .iter()
            .filter(|l| l.starts_with("Program log"))
            .flat_map(|l| {
                let start_ix = l.find(":").map(|ix| ix + 1).unwrap_or(0);
                l[start_ix..].split_whitespace()
            })
            .collect_vec();

        for word in words {
            *map.entry(word.to_owned()).or_insert(0) += 1;
            *map.entry(ALL_KEYWORD.to_owned()).or_insert(0) += 1;
        }
    }
    map
}

fn set_of_parsed_inner_instructions(txs: &[TxInfo]) -> HashSet<String> {
    let mut res = HashSet::new();
    for tx in txs {
        for ins in tx.inner_parsed_instructions.iter() {
            res.insert(parsed_inner_instruction_tag(ins));
        }
    }
    res
}

fn parsed_inner_instruction_tag(ins: &ParsedInstruction) -> String {
    let ins_type = ins
        .parsed
        .get("type")
        .unwrap_or(&serde_json::json!("UNKNOWN"))
        .to_string();
    format!("{}::{}", ins.program, ins_type)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProgramCMP {
    program_addr: String,
    program_instruction_size: usize,
    program_instruction_size_factor: f64,
    program_owned_accounts_cnt: usize,
    program_owned_accounts_avg_size: usize,
    program_owned_accounts_avg_size_factor: f64,
    program_owned_accounts_size_distribution: HashMap<usize, usize>,
    instruction_shapes: usize,
    shape_hits: usize,
    mean_levenshtein_dist_per_shape: Vec<f64>,
    variance_levenshtein_dist_per_shape: Vec<f64>,
    mean_sorensen_dice_per_shape: Vec<f64>,
    variance_sorensen_dice_per_shape: Vec<f64>,
    overlapping_parsed_ins: Vec<String>,
    overlapping_parsed_ins_factor_referent: f64,
    overlapping_parsed_ins_factor_current: f64,
    words_frequency_score: f64,
}

fn cmp_tx_logs(tx1: &TxInfo, tx2: &TxInfo) -> Option<(usize, f64, f64)> {
    if tx1.logs.is_none() || tx2.logs.is_none() {
        return None;
    }

    let l1: Vec<_> = tx1
        .logs
        .as_ref()
        .unwrap()
        .iter()
        .filter(|l| l.starts_with("Program log"))
        .map(|l| {
            let start_ix = l.find(":").map(|ix| ix + 1).unwrap_or(0);
            l[start_ix..].to_owned()
        })
        .collect();
    let l2: Vec<_> = tx2
        .logs
        .as_ref()
        .unwrap()
        .iter()
        .filter(|l| l.starts_with("Program log"))
        .map(|l| {
            let start_ix = l.find(":").map(|ix| ix + 1).unwrap_or(0);
            l[start_ix..].to_owned()
        })
        .collect();
    log::debug!("l1: {:#?}", l1);
    log::debug!("l2: {:#?}", l2);

    let mut s_d_all = 0.0;
    let mut leven_all = 0;
    let mut leven_factor_all = 0.0;

    for ix in 0..std::cmp::min(l1.len(), l2.len()) {
        let l1_line = &l1[ix];
        let l2_line = &l2[ix];

        let s_d_line = strsim::sorensen_dice(l1_line, l2_line);

        let levenshtein_line =
            itertools::zip(l1_line.split_whitespace(), l2_line.split_whitespace())
                .map(|(w1, w2)| strsim::levenshtein(w1, w2))
                .sum::<usize>();

        let chars_in_line = itertools::zip(l1_line.split_whitespace(), l2_line.split_whitespace())
            .map(|(w1, w2)| std::cmp::max(w1.chars().count(), w2.chars().count()))
            .sum::<usize>();

        let leven_factor: f64 = 1.0 - levenshtein_line as f64 / chars_in_line as f64;

        s_d_all += s_d_line;
        leven_all += levenshtein_line;
        leven_factor_all += leven_factor;
    }
    if leven_all == 0 && s_d_all <= 0.1 {
        return None;
    }
    Some((leven_all, s_d_all, leven_factor_all))
}
