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
use reqwest::blocking::Client;
use retry::{delay::Fixed, OperationResult};
use serde::{
    de::{SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use solana_account_decoder::{UiAccountEncoding, UiDataSliceConfig};
use solana_client::{
    rpc_client::RpcClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig},
    rpc_response::RpcConfirmedTransactionStatusWithSignature,
};
use solana_program::{blake3::Hash, pubkey::Pubkey};

use anyhow::{Context, Result};
use solana_sdk::{account::Account, signature::Signature};
use solana_transaction_status::{
    parse_accounts::ParsedAccount, parse_instruction::ParsedInstruction,
    EncodedConfirmedTransactionWithStatusMeta, UiInnerInstructions,
};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const STD_RETRY: u64 = 50;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Turn debugging information on
    #[clap(short, long, parse(from_occurrences))]
    debug: usize,

    /// Solana RPC url to use. This is optional and will use mainnet by default.
    /// Note that A LOT of requests will be issued to the RPC, so it is highly advisable
    /// to use a dedicated, private RPC node, with higher rate limit then the public one
    /// when using the analyze command.
    #[clap(short, long)]
    rpc_url: Option<String>,

    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// List program addresses from the chain and puts them into output file.
    /// This output is useful as an entry command for `analyze` command.
    ListPrograms {
        /// Sets an output file
        #[clap(short, long, parse(from_os_str))]
        output: PathBuf,

        /// Should program addresses that are not executable be printed
        #[clap(short, long)]
        print_non_executable: bool,
    },

    /// Analyzes the program addresses specified in the input file, and
    /// compares them to the program address given as a `referent_addr`
    /// to see how similar they are. Generates an output file with
    /// information about the analyzed programs.
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

        /// Transactions per program to analyze. The larger the value
        /// more precise it will be, but it will last longer. Default is 50,
        /// but it is advisable to use order of magnitude larger, especially
        /// for programs with diverse commands.
        #[clap(short, long, default_value_t = 50)]
        tx_cnt: usize,
    },

    /// Uses the output of the `analyze` command as an input to generate
    /// a TOP N report looking at the different tracked metrics. All metrics
    /// are normalized and considered equal.
    PickTop {
        /// Sets an input file to parse and analyze
        #[clap(short, long, parse(from_os_str))]
        input: PathBuf,

        /// Picking top N for each category
        #[clap(short, long, default_value_t = 5)]
        n: usize,
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
        Some(Commands::PickTop { input, n }) => {
            pick_top(input, n)?;
        }
        None => {
            panic!("Command must be specified");
        }
    }

    Ok(())
}

fn pick_top(input: PathBuf, n: usize) -> Result<()> {
    let input: Result<Vec<ProgramCMP>, _> = std::fs::read_to_string(input)?
        .lines()
        .map(|l| serde_json::from_str(l))
        .collect();
    let mut input = input?;

    let mut pids = HashMap::new();

    log::info!("==== 1. addrs by program size factor");
    input.sort_by(|a, b| {
        let mut a_s = a.program_instruction_size_factor;
        let mut b_s = b.program_instruction_size_factor;
        if a_s > 1.0 {
            a_s = 1.0 / a_s;
        }

        if b_s > 1.0 {
            b_s = 1.0 / b_s;
        }

        b_s.partial_cmp(&a_s).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!("{} - {}", p.program_addr, p.program_instruction_size_factor);
    });

    log::info!("==== 2. addrs by program owned accounts size factor");
    input.sort_by(|a, b| {
        let mut a_s = a.program_owned_accounts_avg_size_factor;
        let mut b_s = b.program_owned_accounts_avg_size_factor;
        if a_s > 1.0 {
            a_s = 1.0 / a_s;
        }

        if b_s > 1.0 {
            b_s = 1.0 / b_s;
        }

        b_s.partial_cmp(&a_s).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!(
            "{} - {}",
            p.program_addr,
            p.program_owned_accounts_avg_size_factor
        );
    });

    log::info!("==== 3. addrs by shape hits");
    input.sort_by(|a, b| {
        let a_s = a.shape_hits;
        let b_s = b.shape_hits;
        b_s.partial_cmp(&a_s).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!("{} - {}", p.program_addr, p.shape_hits);
    });

    log::info!("==== 4. addrs by mean levenshtien");
    input.sort_by(|a, b| {
        let a_l_sum: f64 = a.mean_levenshtein_dist_per_shape.iter().sum();
        let b_l_sum: f64 = b.mean_levenshtein_dist_per_shape.iter().sum();
        b_l_sum.partial_cmp(&a_l_sum).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!(
            "{} - {:?}",
            p.program_addr,
            p.mean_levenshtein_dist_per_shape
        );
    });

    log::info!("==== 5. addrs by mean sorensen");
    input.sort_by(|a, b| {
        let a_l_sum: f64 = a.mean_sorensen_dice_per_shape.iter().sum();
        let b_l_sum: f64 = b.mean_sorensen_dice_per_shape.iter().sum();
        b_l_sum.partial_cmp(&a_l_sum).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!("{} - {:?}", p.program_addr, p.mean_sorensen_dice_per_shape);
    });

    log::info!("==== 6. addrs by instructions overlapping with referent and current");
    input.sort_by(|a, b| {
        let a: f64 =
            a.overlapping_parsed_ins_factor_referent + a.overlapping_parsed_ins_factor_current;
        let b: f64 =
            b.overlapping_parsed_ins_factor_referent + b.overlapping_parsed_ins_factor_current;
        b.partial_cmp(&a).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!(
            "{} - {:?}",
            p.program_addr,
            p.overlapping_parsed_ins_factor_referent + p.overlapping_parsed_ins_factor_current
        );
    });

    log::info!("==== 7. addrs by log words frequency score");
    input.sort_by(|a, b| {
        let a: f64 = a.words_frequency_score;
        let b: f64 = b.words_frequency_score;
        b.partial_cmp(&a).unwrap()
    });
    input.iter().take(n).for_each(|p| {
        *pids.entry(p.program_addr.to_owned()).or_insert(0) += 1;
        log::info!("{} - {:?}", p.program_addr, p.words_frequency_score);
    });

    let mut pids = pids.into_iter().map(|(k, v)| (k, v)).collect_vec();
    pids.sort_by(|a, b| b.1.cmp(&a.1));

    log::info!("TOP candidates are:");
    for (addr, v) in pids {
        log::info!("{} - {:?}", addr, v);
    }

    Ok(())
}

fn analyze_one(
    rpc: &RpcClient,
    http_client: &Client,
    addr: String,
    tx_cnt: usize,
) -> Result<ProgramInfo> {
    log::info!("Fetching info for program on addr={}", addr);
    let key = Pubkey::from_str(&addr)?;
    let program_acc = retry_n(STD_RETRY, || rpc.get_account(&key))?;

    if !program_acc.executable {
        return Err(anyhow::anyhow!("Program not executable!"));
    }

    let mut last_deployed_timestamp = 0 as i64;
    let program_size;
    use solana_program::bpf_loader_upgradeable::UpgradeableLoaderState;
    if let Ok(UpgradeableLoaderState::Program {
        programdata_address,
    }) = program_acc.deserialize_data()
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
    } else {
        // this is not upgradable, so program data is embedded into normal program address
        program_size = program_acc.data.len();
    }

    let size_occurencies = retry_n(STD_RETRY, || {
        extract_size_distributions(&http_client, &rpc.url(), &addr)
    })?;

    log::info!(
        "Extracted size distributions for {} owned accounts",
        size_occurencies.iter().map(|(k, v)| *v).sum::<usize>()
    );

    let sigs = get_latest_tx_sigs(rpc, &key, tx_cnt)?;

    log::info!(
        "Extracting tx info for {} transactions for addr {}",
        sigs.len(),
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

/// Instead of using solana RPC client to perform `get_program_accounts` call, here it is
/// implemented manually, since this endpoint doesn't have pagination implemented, and allocates
/// a huge amounts of memory when interacting with programs that own a lot of accounts.
///
/// This implementation reduces json response from the data stream into map of size distributions,
/// instead of buffering the vector of response objects in the memory, and calculating distribution
/// afterwards.
///
/// If needed in the future, this method can be generalized to accept a visitor that can perform
/// some other reduction operation if needed.
fn extract_size_distributions(
    http_client: &Client,
    rpc_url: &str,
    program_addr: &str,
) -> std::result::Result<HashMap<usize, usize>, solana_client::client_error::ClientError> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method":"getProgramAccounts",
        "params": [ program_addr, { "encoding": "base64" } ]
    });

    let resp = http_client
        .post(rpc_url)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap())
        .send()
        .map_err(|e| {
            let e_kind = solana_client::client_error::ClientErrorKind::Custom(format!(
                "Error while communicating with solana RPC endpoint - {:?}",
                e
            ));
            solana_client::client_error::ClientError {
                request: None,
                kind: e_kind,
            }
        })?;

    use std::io::BufReader;
    // buffering here is important for performance reasons, hyper doesn't buffer response by default.
    let buff = BufReader::with_capacity(8 * 1024, resp);
    let res: RpcGetProgramAccsResp = serde_json::from_reader(buff).map_err(|e| {
        let e_kind = solana_client::client_error::ClientErrorKind::Custom(format!(
            "Error while deserializing resp from `getProgramAccounts` - {:?}",
            e
        ));
        solana_client::client_error::ClientError {
            request: None,
            kind: e_kind,
        }
    })?;

    Ok(res.size_distributions)
}

#[derive(Deserialize, Debug)]
struct RpcGetProgramAccsResp {
    jsonrpc: String,

    #[serde(deserialize_with = "size_distributions")]
    #[serde(rename(deserialize = "result"))]
    size_distributions: HashMap<usize, usize>,
}

fn size_distributions<'de, D>(deserializer: D) -> Result<HashMap<usize, usize>, D::Error>
where
    D: Deserializer<'de>,
{
    struct DistributionsVisitor();

    impl<'de> Visitor<'de> for DistributionsVisitor {
        type Value = HashMap<usize, usize>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("Unexpected json input data format")
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<HashMap<usize, usize>, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut size_distributions: HashMap<usize, usize> = HashMap::new();

            while let Some(value) = seq.next_element::<AccInfo>()? {
                let data_size = value.account.data[0].len();
                *size_distributions.entry(data_size).or_insert(0) += 1;
            }

            Ok(size_distributions)
        }
    }

    let visitor = DistributionsVisitor();
    deserializer.deserialize_seq(visitor)
}

#[derive(Deserialize)]
struct AccInfo {
    account: AccData,
    pubkey: String,
}

#[derive(Deserialize)]
struct AccData {
    data: [String; 2],
    executable: bool,
    lamports: u64,
    owner: String,
    #[serde(rename(deserialize = "rentEpoch"))]
    rent_epoch: u64,
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

const LIST_SIGNATURES_PAGE_LIMIT: usize = 1000;

fn get_latest_tx_sigs(
    rpc: &RpcClient,
    addr: &Pubkey,
    tx_cnt: usize,
) -> Result<Vec<RpcConfirmedTransactionStatusWithSignature>> {
    let mut results = Vec::with_capacity(tx_cnt);
    let mut count = 0;
    let mut before: Option<Signature> = None;

    while tx_cnt > count {
        let cur_limit = if tx_cnt > LIST_SIGNATURES_PAGE_LIMIT + count {
            LIST_SIGNATURES_PAGE_LIMIT
        } else {
            tx_cnt - count
        };
        let sigs = retry_n(STD_RETRY, || {
            let cfg = solana_client::rpc_client::GetConfirmedSignaturesForAddress2Config {
                limit: Some(cur_limit),
                before,
                ..Default::default()
            };
            rpc.get_signatures_for_address_with_config(&addr, cfg)
        })?;

        if let Some(last) = sigs.last() {
            before = Some(last.signature.parse()?);
            count += sigs.len();
            results.extend(sigs);
        } else {
            // We didn't find required number of signatures, but there are no more for this Addr
            // since we received an empty page while listing.
            break;
        }
    }

    log::info!("Found {} tx signatures for addr {}", results.len(), addr);

    Ok(results)
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
            let inner_inst: Option<Vec<UiInnerInstructions>> =
                Option::from(tx_meta.inner_instructions);
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
                logs: Option::from(tx_meta.log_messages),
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
    let config = RpcProgramAccountsConfig {
        filters: None,
        account_config: RpcAccountInfoConfig {
            encoding: Some(UiAccountEncoding::Base64),
            data_slice: Some(UiDataSliceConfig {
                offset: 0,
                length: 0,
            }),
            commitment: None,
            min_context_slot: None,
        },
        with_context: Some(false),
    };
    let res = rpc
        .get_program_accounts_with_config(&key, config)
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
    // TODO make threads configurable?
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(8, tx_cnt / 5))
        .build_global()
        .unwrap();

    let http_client = reqwest::blocking::Client::new();

    let mut out = std::fs::File::create(&output).with_context(|| {
        format!(
            "Unable to create output file on path `{}`",
            output.to_string_lossy()
        )
    })?;

    let referent_info = analyze_one(
        &rpc,
        &http_client,
        referent_addr,
        std::cmp::min(4 * tx_cnt, 5000),
    )?;

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

    for addr in std::fs::read_to_string(input)?.lines() {
        let mut err_cnt = 0;
        let other_info = loop {
            let info = analyze_one(&rpc, &http_client, addr.to_owned(), tx_cnt);

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
                "Dropping program address `{}` due to error: {:?}",
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
        let cmp = cmp_two_programs(tx_cnt, &referent_info, &other_info);

        log::info!("Compare finished - {:?}", cmp);
        writeln!(&mut out, "{}", serde_json::to_string(&cmp)?)?;
    }

    out.flush().context("Error while flushing output!")?;

    Ok(())
}

fn cmp_two_programs(tx_cnt: usize, p1: &ProgramInfo, p2: &ProgramInfo) -> ProgramCMP {
    let p1_tx_map = p1.tx_infos_by_shape();
    let p2_tx_map = p2.tx_infos_by_shape();

    let mut shape_hits = 0usize;
    let mut mean_levenshtein_dist_per_shape = vec![];
    let mut variance_levenshtein_dist_per_shape = vec![];

    let mut mean_sorensen_dice_per_shape = vec![];
    let mut variance_sorensen_dice_per_shape = vec![];

    // limiting the cartesian product to since I don't want it to go overboard
    // 10k might even be too much
    // TODO maybe make this configurable input param of the CLI
    let mut limit = tx_cnt * tx_cnt;
    if limit > 10_000 {
        limit = 10_000;
    }

    for p1_shape in p1_tx_map {
        if p2_tx_map.contains_key(&p1_shape.0) {
            shape_hits += 1;
            let tx1s = &p1_shape.1;
            let tx2s = &p2_tx_map[&p1_shape.0];

            let tx_pairs = tx1s
                .iter()
                .cartesian_product(tx2s.iter())
                .take(limit)
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
    let overlapping_parsed_ins_factor_referent = if p1.set_of_parsed_ins.len() == 0 {
        0.0
    } else {
        overlapping_parsed_ins.len() as f64 / p1.set_of_parsed_ins.len() as f64
    };
    let overlapping_parsed_ins_factor_current = if p2.set_of_parsed_ins.len() == 0 {
        0.0
    } else {
        overlapping_parsed_ins.len() as f64 / p2.set_of_parsed_ins.len() as f64
    };

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

    let program_owned_accounts_avg_size_factor = if p2.program_owned_accounts_avg_size() == 0 {
        0.0
    } else {
        p1.program_owned_accounts_avg_size() as f64 / p2.program_owned_accounts_avg_size() as f64
    };

    ProgramCMP {
        program_addr: p2.addr.clone(),
        program_last_deployed_ts: Some(p2.last_deployed_ts),
        program_instruction_size: p2.program_size,
        program_instruction_size_factor: p1.program_size as f64 / p2.program_size as f64,
        program_owned_accounts_cnt: p2.program_owned_accounts_cnt(),
        program_owned_accounts_avg_size: p2.program_owned_accounts_avg_size(),
        program_owned_accounts_avg_size_factor,
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
        let logs = tx.logs.as_ref();
        if logs.is_none() {
            continue;
        }

        let words = logs
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
    program_last_deployed_ts: Option<i64>,
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
