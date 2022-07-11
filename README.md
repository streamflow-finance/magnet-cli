### Magnet CLI

CLI tool that analyses Solana programs in order to check how similar they are to the referent program (are they fork / redeployment).

## Usage guide

CLI has 3 commands that feed into each other, and should be used sequentially.
```
USAGE:
    magnet-cli [OPTIONS] [SUBCOMMAND]

OPTIONS:
    -d, --debug                Turn debugging information on
    -h, --help                 Print help information
    -r, --rpc-url <RPC_URL>    Solana RPC url to use. This is optional and will use mainnet by
                               default. Note that A LOT of requests will be issued to the RPC, so it
                               is highly advisable to use a dedicated, private RPC node, with higher
                               rate limit then the public one when using the analyze command
    -V, --version              Print version information

SUBCOMMANDS:
    list-programs    List program addresses from the chain and puts them into output file. This
                         output is useful as an entry command for `analyze` command
    analyze          Analyzes the program addresses specified in the input file, and compares
                         them to the program address given as a `referent_addr` to see how similar
                         they are. Generates an output file with information about the analyzed
                         programs
    help             Print this message or the help of the given subcommand(s)
    
    pick-top         Uses the output of the `analyze` command as an input to generate a TOP N
                         report looking at the different tracked metrics. All metrics are normalized
                         and considered equal
```

First command that should be executed is `list-programs` command:

```
List program addresses from the chain and puts them into output file. This output is useful as an
entry command for `analyze` command

USAGE:
    magnet-cli list-programs [OPTIONS] --output <OUTPUT>

OPTIONS:
    -h, --help                    Print help information
    -o, --output <OUTPUT>         Sets an output file
    -p, --print-non-executable    Should program addresses that are not executable be printed
```

This command will fetch all of the executable programs from the chain (those owned by any of the BPF loaders), and will print their program addresses in the specified output file, one address per line. This file could be then sliced/filtered by user of the cli before it feeds into the next command - `analyze`.

```
Analyzes the program addresses specified in the input file, and compares them to the program address
given as a `referent_addr` to see how similar they are. Generates an output file with information
about the analyzed programs

USAGE:
    magnet-cli analyze [OPTIONS] --input <INPUT> --output <OUTPUT> --referent-addr <REFERENT_ADDR>

OPTIONS:
    -h, --help
            Print help information

    -i, --input <INPUT>
            Sets an input file

    -o, --output <OUTPUT>
            Sets an output file

    -r, --referent-addr <REFERENT_ADDR>
            Referent program address

    -t, --tx-cnt <TX_CNT>
            Transactions per program to analyze. The larger the value more precise it will be, but
            it will last longer. Default is 50, but it is advisable to use order of magnitude
            larger, especially for programs with diverse commands [default: 50]
```

This command will analyze the transaction data from the programs provided in the input file. For each program address, it will fetch a number of transactions specified by the flag of this command (default is 50), and crunch the information comparing the transaction data of each program with the specified referent program looking for similarities. Results of the analyses will be printed into output file, one json per line (one per program address), containing information about the different metrics that were tracked for the program. Those things are:

- program_addr: Address of the program
- program_last_deployed_ts: If available timestamp when the program was last deployed
- program_instruction_size: Size of the program
- program_instruction_size_factor: Factor of the program size compared to the referent program
- program_owned_accounts_cnt: How many accounts are owned by this program
- program_owned_accounts_avg_size: Average size of the account owned by this program
- program_owned_accounts_avg_size_factor: Factor of the average owned account size compared to the referent
- program_owned_accounts_size_distribution: Distribution of sizes for owned accounts
- instruction_shapes: Number of different "shapes" of instructions executed by the program, looking at the supplied accounts
- shape_hits: How many shapes are similar to the referent program shapes
- mean_levenshtein_dist_per_shape: Mean levenshtein distance between the log lines of the current program compared to the referent program
- variance_levenshtein_dist_per_shape: Variance of the above metric
- mean_sorensen_dice_per_shape: Mean Sorensen similarity index between the log lines of the current program compared to the referent program
- variance_sorensen_dice_per_shape: Variance of the above metric
- overlapping_parsed_ins: Parsed program instructions occurring in both this program and referent program
- overlapping_parsed_ins_factor_referent: How much are overlapping instructions covering the referent program instructions
- overlapping_parsed_ins_factor_current: How much are overlapping instructions covering the current program instructions
- words_frequency_score: Score of frequency of words occurring in the log lines of referent vs current program

This output will contain json for each program, but it is probably easier to analyze it with a `pick-top` command then manually. But you can still remove some
outliers and double check the data manually. So in the end, `pick-top` command is the last one:

```
Uses the output of the `analyze` command as an input to generate a TOP N report looking at the
different tracked metrics. All metrics are normalized and considered equal

USAGE:
    magnet-cli pick-top [OPTIONS] --input <INPUT>

OPTIONS:
    -h, --help             Print help information
    -i, --input <INPUT>    Sets an input file to parse and analyze
    -n, --n <N>            Picking top N for each category [default: 5]
```
This will parse the input file generated as an output of the analyze command, and will generate a report sorted by each metric that was tracked with the analyze command.
In the end it will print out the list of top candidates, programs that were in the TOP N for the largest number of categories. You can consider only the last part of the command, but it is advisable to look into individual metrics, especially if you know that some are more important then others for the referent program you are analyzing.

## Performance and rate limits

Note that `analyze` command can execute for a very long time. For each program address, we are fetching the number of transaction data from the chain, and that is one RPC request per transaction (default 50). If used with large `tx-cnt` supplied (and the larger the better from the precision standpoint), number of requests against the solana RPC will be scaled accordingly. If you do not use a private RPC node, with higher rate limit, it is not advisable to go beyond the tx-cnt of 50 because the execution will be smothered by the rate limits. CLI uses fantastic `rayon` library to parallelize the transaction fetching and calculations.

## Example usage

Here is one example how CLI can be used

```
// execute the list command
./magnet-cli list-programs --output out/test

// use only the first 10 program addresses, since this is for demo, never do this in actual usage
head -n10 out/test > out/test-first-10

// execute the analyze command
./magnet-cli --rpc-url <PRIVATE_RPC_URL> analyze --input out/test-first-10 \
                                                 --output out/analyze-output-test-first-10 \
                                                 --referent-addr strmRqUCoQUgGUan5YhzUZa6KqdzwX5L6FpUxfmKg5m \
                                                 --tx-cnt 100

// then execute the pick top command to find the best matches
./magnet-cli pick-top --input out/analyze-output-test-first-10 -n 3
```

## Limitations

Use with caution, hand pick and double check the best results to see if it makes sense. This tool can help find the needles in the haystack, but it is not 100% correct, and if there are no forks at all it will probably spit out some unrelated results. Always use the referent program in the input list as a control group.

