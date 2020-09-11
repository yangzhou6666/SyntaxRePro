#! /bin/sh

if [ ! -d error_recovery_experiment ]; then
    git clone https://github.com/softdevteam/error_recovery_experiment.git
fi


cd error_recovery_experiment

cd runner

set -e

GRMTOOLSV=grmtools-0.7.0
GRAMMARSV=926274486b2e81c78cf41faa6a600e62bd788772

if [ ! -d grmtools ]; then
    git clone https://github.com/softdevteam/grmtools
    cd grmtools
    git checkout ${GRMTOOLSV}
    cd ..
fi

if [ ! -d grammars ]; then
    git clone https://github.com/softdevteam/grammars/
    cd grammars && git checkout ${GRAMMARSV}
    cd ..
    cp grammars/java7/java.l java_parser/src/java7.l
    cp grammars/java7/java.y java_parser/src/java7.y
fi

cd grmtools/nimbleparse

cargo build --release

cd ../target/release

cp nimbleparse ../../../../../

