#!sh

# for expname in dwn_6_0 dwn_6_0__dedup dwn_6_3 dwn_6_3__dedup original; do
# for expname in dedup; do
for expname in dwn_6_0__dedup dwn_6_3__dedup; do
    if [ ! -d DataFairseqEncoded/$expname ]; then
        mkdir -p DataFairseqEncoded/$expname;
    fi

    TEXT=DataEncoded/$expname/data
    fairseq-preprocess \
        --source-lang unit.bpe \
        --target-lang eng.bpe \
            --trainpref $TEXT/train \
            --validpref $TEXT/dev \
            ` # --testpref $TEXT/test ` \
        --destdir DataFairseqEncoded/$expname/data-bin/unit-eng \
        --workers 8
        # --workers 20
done
