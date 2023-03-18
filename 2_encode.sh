#!sh
for expname in dedup dwn_6_0 dwn_6_0__dedup dwn_6_3 dwn_6_3__dedup original; do
    if [ ! -d DataEncoded/$expname ]; then
        mkdir DataEncoded/$expname;
    fi

    if [ ! -d DataEncoded/$expname/table ]; then
        mkdir DataEncoded/$expname/table;
    fi

    if [ ! -d DataEncoded/$expname/data ]; then
        mkdir DataEncoded/$expname/data;
    fi

    python3 fairseq_exp/subword-nmt/learn_bpe.py \
        -s 10000 \
            < "DataProcessed/$expname/train.eng" \
            > DataEncoded/$expname/table/code_for_eng
        
    # ~~~~~~~~~~~~~~~~``        
    head -n 2000 "DataProcessed/$expname/train.unit" \
        > "DataProcessed/$expname/train.unit_head"
    # ~~~~~~~~~~~~~~~~``__________
        
    python3 fairseq_exp/subword-nmt/learn_bpe.py \
        -s 10000 \
            < "DataProcessed/$expname/train.unit_head" \
            > DataEncoded/$expname/table/code_for_unit
    # ~~~~~~~~~~~~~~~~``        
    rm -f "DataProcessed/$expname/train.unit_head"
    # ~~~~~~~~~~~~~~~~``__________

    python3 fairseq_exp/subword-nmt/apply_bpe.py \
        -c DataEncoded/$expname/table/code_for_unit \
            < DataProcessed/$expname/train.unit \
            > DataEncoded/$expname/data/train.unit.bpe
    echo 'Encoded! 1___'
    python3 fairseq_exp/subword-nmt/apply_bpe.py \
        -c DataEncoded/$expname/table/code_for_eng \
            < DataProcessed/$expname/train.eng \
            > DataEncoded/$expname/data/train.eng.bpe
    echo 'Encoded! _2__'
    python3 fairseq_exp/subword-nmt/apply_bpe.py \
        -c DataEncoded/$expname/table/code_for_unit \
            < DataProcessed/$expname/dev.unit \
            > DataEncoded/$expname/data/dev.unit.bpe
    echo 'Encoded! __3_'
    python3 fairseq_exp/subword-nmt/apply_bpe.py \
        -c DataEncoded/$expname/table/code_for_eng \
            < DataProcessed/$expname/dev.eng \
            > DataEncoded/$expname/data/dev.eng.bpe
    echo 'Encoded! ___4'
done
