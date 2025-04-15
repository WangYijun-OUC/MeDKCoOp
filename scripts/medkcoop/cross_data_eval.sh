for SEED in 1 2 3
do
    bash scripts/medkcoop/xd_test.sh CHMNIST ${SEED}
    bash scripts/medkcoop/xd_test.sh covid ${SEED}
    bash scripts/medkcoop/xd_test.sh ctkidney ${SEED}
    bash scripts/medkcoop/xd_test.sh lungcolon ${SEED}
    bash scripts/medkcoop/xd_test.sh retina ${SEED}
    bash scripts/medkcoop/xd_test.sh octmnist ${SEED}
    bash scripts/medkcoop/xd_test.sh kvasir ${SEED}
done