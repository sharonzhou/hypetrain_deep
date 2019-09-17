# Starter
Starter repository. This is modeled after the initial aicc-spr19-wind repo, and works with the baseline model there :)

## Activate environment
`source activate <ENV_NAME>`

## Train Usage

`python train.py --experiment_name <name of experiment>`

## Test usage
Single model:

`python test.py --ckpt_path <path to checkpoint> --phase {valid, test}`

Ensemble:

`python test.py --config_path <path to config> --phase {valid, test}`

