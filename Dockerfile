FROM rust:buster as build

# create a new empty shell project
RUN USER=root cargo new --bin magnet-cli
WORKDIR /magnet-cli

RUN apt-get update && apt-get install libudev-dev

# copy over your manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

# this build step will cache your dependencies
RUN cargo build --release
RUN rm src/*.rs

# copy your source tree
COPY ./src ./src

# build for release
RUN rm ./target/release/deps/magnet_cli*
RUN cargo build --release

# our final base
FROM debian:buster-slim

RUN apt-get update && apt-get install libudev-dev

# copy the build artifact from the build stage
COPY --from=build /magnet-cli/target/release/magnet-cli .

# set the startup command to run your binary
CMD ["./magnet-cli"]
