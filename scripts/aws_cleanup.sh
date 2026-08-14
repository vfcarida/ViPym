#!/usr/bin/env bash
# ViPym Ephemeral AWS Cluster Cleanup Script
# Terminates any standing EC2 instances tagged with ViPymExperiment and cleans up temp resources

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
TAG_KEY="Project"
TAG_VALUE="ViPym"

echo "=== Searching for active ViPym EC2 instances in ${AWS_REGION} ==="
INSTANCE_IDS=$(aws ec2 describe-instances \
    --region "${AWS_REGION}" \
    --filters "Name=tag:${TAG_KEY},Values=${TAG_VALUE}" "Name=instance-state-name,Values=running,pending,stopped" \
    --query "Reservations[*].Instances[*].InstanceId" \
    --output text)

if [ -z "${INSTANCE_IDS}" ] || [ "${INSTANCE_IDS}" == "None" ]; then
    echo "✓ No active ViPym EC2 instances found."
else
    echo "Terminating instances: ${INSTANCE_IDS}"
    aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids ${INSTANCE_IDS}
    echo "✓ Termination requested."
fi

echo "=== ViPym Cloud Cleanup Completed ==="
