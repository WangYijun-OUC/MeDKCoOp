for SEED in 1 2 3
do
    bash scripts/medkcoop/xd_test.sh CHMNISTv2 ${SEED}
    bash scripts/medkcoop/xd_test.sh covidv2 ${SEED}
    bash scripts/medkcoop/xd_test.sh ctkidneyv2 ${SEED}
    bash scripts/medkcoop/xd_test.sh lungcolonv2 ${SEED}
    bash scripts/medkcoop/xd_test.sh retinav2 ${SEED}
    bash scripts/medkcoop/xd_test.sh octmnistv2 ${SEED}
    bash scripts/medkcoop/xd_test.sh kvasirv2 ${SEED}
done