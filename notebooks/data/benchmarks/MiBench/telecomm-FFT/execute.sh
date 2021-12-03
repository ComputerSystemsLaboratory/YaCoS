#!/bin/bash

function execute() {
    case $TOOL in
        "perf")
            PERF_TOOL="cycles instructions"
            PERF_TYPE="u"

            events=${PERF_TOOL// /:$PERF_TYPE,}
            events=$events:$PERF_TYPE

            if [[ $WARMUP_CACHE -eq 1 ]]; then
                timeout --signal=TERM ${RUNTIME} ./a.out $RUN_OPTIONS < $STDIN &> /dev/null
                if [[ $? -ne 0 ]]; then
                    echo "Halting warmup cache due to some error" > error.log
                    exit 1
                fi
            fi

            timeout --signal=TERM ${RUNTIME} perf stat -x "," -r \
                    $TIMES -o runtime.csv -e $events \
                    bash -c "./a.out $RUN_OPTIONS < $STDIN" &> output.txt
            if [[ $? -ne 0 ]]; then
                echo "Halting execution (perf) due to some error" > error.log
                exit 1
            fi
            data=`sed '1!d' runtime.csv | awk -F',' '{printf "%s", $1}'` ; echo "- $data" > cycles.yaml
            data=`sed '2!d' runtime.csv | awk -F',' '{printf "%s", $1}'` ; echo "- $data" > instructions.yaml
            ;;
        "hyperfine")
            hyperfine -w $WARMUP_CACHE -r $TIMES --show-output \
                      --export-csv runtime.csv \
                      -u second "./a.out $RUN_OPTIONS < $STDIN" &> output.txt
            if [[ $? -ne 0 ]]; then
                echo "Halting execution (hyperfine) due to some error" > error.log
                exit 1
            fi
            data=`sed '2!d' runtime.csv | awk -F',' '{printf "%s", $2}'` ; echo "- $data" > runtime.yaml
            ;;
        *)
            echo "Error: this tool is not implemented yet" > error.log
            exit 1
            ;;
    esac
}

function verify_output() {
    case $TOOL in
        "perf")
            # Remove duplicates
            cp output.txt output.all
            lines=`wc output.txt | awk '{print $1}'`
            lines=$(($lines/$TIMES))
            head -n $lines output.all | cat -v > output.txt
            ;;
        "hyperfine")
            # Verify warning
            grep "Warning:" output.txt /dev/null
            if [[ $? -eq 0 ]]; then
                WARNING=1
            else
                WARNING=0
            fi
            # Fix output.txt
            lines=`wc output.txt | awk '{print $1}'`
            head -n $(($lines - $WARNING - 3)) output.txt > output.txt.1
            tail -n $(($lines - $WARNING - 4)) output.txt.1 > output.txt.2
            lines=$(($(($lines - $WARNING - 4))/$(($TIMES+$WARMUP_CACHE))))
            head -n $lines output.txt.2 | cat -v > output.txt
            rm -f output.txt.1 output.txt.2
            ;;
        *)
            echo "Error: this tool is not implemented yet" > error.log
            exit 1
            ;;
    esac

    # Diff the two files.
    diff reference_output/reference_output_$WORKING_SET.txt output.txt > diff.txt 2>&1
    if [[ $? -eq 0 ]]; then
        # They are igual
        echo "succeed" > verify_output.yaml
    else
        # They are different
        echo "failed" > verify_output.yaml
    fi

}

# Command line parameters
WORKING_SET=$1
TIMES=$2
TOOL=$3
VERIFY_OUTPUT=$4
WARMUP_CACHE=$5
RUNTIME=$6
STDIN=/dev/null

case $WORKING_SET in
    0)
        RUN_OPTIONS="4 8192 -i"
        ;;
    1)
        RUN_OPTIONS="8 32768 -i"
        ;;
    2)
        RUN_OPTIONS="16 65536 -i"
        ;;
    3)
        RUN_OPTIONS="24 131072 -i"
        ;;
    4)
        RUN_OPTIONS="32 262144 -i"
        ;;
    *)
        echo "Error: dataset"
        exit 1
	;;
esac

execute

if [[ $VERIFY_OUTPUT -eq 1 ]]; then
   verify_output
fi
