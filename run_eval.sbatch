#!/bin/bash
#SBATCH --job-name=svbench
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=${SV_DIR:-$PWD}/slurm/%x-%j.out
#SBATCH --error=${SV_DIR:-$PWD}/slurm/%x-%j.err
# Usage: sbatch [--qos=...] --export=ALL,MODEL=<path>,LABEL=<name>[,LIMIT=n][,SQUAD_LIMIT=n] run_eval.sbatch
set -u
: "${MODEL:?}"; : "${LABEL:?}"
SV=${SV_DIR:-$PWD}
SCR=$SCRATCH_DIR
SIF=$EVAL_SIF
OUT=$SV/results/$LABEL${LIMIT:+-limit$LIMIT}
mkdir -p "$OUT" "$SV/slurm"
export HF_HOME=$SCR/oellm-eval-hf HF_DATASETS_CACHE=$SCR/oellm-eval-hf/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SV"
echo "host=$(hostname) job=$SLURM_JOB_ID model=$MODEL limit=${LIMIT:-full} squad_limit=${SQUAD_LIMIT:-1000}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {  # $1 tasks, $2 extra args...
  tasks=$1; shift
  singularity exec --nv --bind "$SV:$SV" --bind "$SCR:$SCR" \
    --env HF_HOME=$HF_HOME --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python -m lm_eval --model hf \
      --model_args "pretrained=$MODEL,dtype=bfloat16,add_bos_token=False" \
      --include_path "$SV/tasks" --tasks "$tasks" --num_fewshot 5 \
      --batch_size auto --output_path "$OUT/results.json" --log_samples "$@"
}

MC="arc_challenge_sv_mcf_svbv2,belebele_swe_Latn_mcf_svbv2,goldenswag_mt_sv_cf_svbv2,ogx_truthfulqax_mc1_sv_svbv2,scandisent_sv_mcf_svbv2,sib200_sv_mcf_svbv2"
V1="finbench_analogies_sv_mcf_svbv2,finbench_emotions_1k_sv_mcf_svbv2,finbench_general_knowledge_sv_mcf_svbv2,finbench_hhh_alignment_sv_mcf_svbv2,finbench_similarities_abstraction_sv_mcf_svbv2"
echo "=== stage 1: multiple-choice tasks"; run "$MC,$V1" ${LIMIT:+--limit $LIMIT}; s1=$?
echo "=== stage 2: squad (generation)";    run squad_sv_gen_svbv2 --limit "${SQUAD_LIMIT:-1000}"; s2=$?
echo "exit stage1=$s1 stage2=$s2"; exit $((s1 + s2))
