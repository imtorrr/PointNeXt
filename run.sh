#!/bin/bash

# Colors for fancy output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print fancy banner
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================"
    echo "   ___  ___ _____  _____   ___ _____ ___   _____ ___  "
    echo "  / _ \/   \\_   _|/ _ \\ \\ / /_   _|_ _| |_   _/ _ \\ "
    echo " | | | | | || || | | | \\ V /  | |  | || __|| || | | |"
    echo " | |_| | |_|| || | |_| || |   | |  | || |_ | || |_| |"
    echo "  \\___/|___/|_||_|\\___/ |_|   |_| |___||__||_| \\___/ "
    echo "                                                      "
    echo "         P O I N T C L O U D   T R A I N I N G        "
    echo "================================================================"
    echo -e "${NC}"
}

# Function to list available configs
list_configs() {
    echo -e "${YELLOW}${BOLD}Available Configuration Files:${NC}\n"

    local configs=($(find cfgs -name "*.yaml" -type f | sort))
    local i=1

    for config in "${configs[@]}"; do
        echo -e "${GREEN}[$i]${NC} $config"
        ((i++))
    done

    echo -e "\n${CYAN}[C]${NC} Enter custom path"
    echo ""
}

# Function to get user config selection
get_config_choice() {
    local configs=($(find cfgs -name "*.yaml" -type f | sort))
    local config_path=""

    while true; do
        echo -e "${BOLD}Select a configuration file (enter number or 'C' for custom):${NC} " >&2
        read -r choice

        if [[ "$choice" == "C" ]] || [[ "$choice" == "c" ]]; then
            echo -e "${BOLD}Enter the full path to your config file:${NC} " >&2
            read -r custom_path
            if [[ -f "$custom_path" ]]; then
                config_path="$custom_path"
                break
            else
                echo -e "${RED}Error: File not found. Please try again.${NC}\n" >&2
            fi
        elif [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -le "${#configs[@]}" ]]; then
            config_path="${configs[$((choice-1))]}"
            break
        else
            echo -e "${RED}Invalid choice. Please enter a valid number or 'C'.${NC}\n" >&2
        fi
    done

    echo "$config_path"
}

# Function to extract metrics from log file
extract_metrics() {
    local log_file="$1"

    # Extract best validation metrics
    local best_epoch=$(grep -oP "Best ckpt @E\K\d+" "$log_file" | tail -1)
    local val_miou=$(grep -oP "val_miou \K[\d.]+" "$log_file" | tail -1)
    local val_macc=$(grep -oP "val_macc \K[\d.]+" "$log_file" | tail -1)
    local val_oa=$(grep -oP "val_oa \K[\d.]+" "$log_file" | tail -1)

    # Extract test metrics if available
    local test_miou=$(grep -oP "test_miou: \K[\d.]+" "$log_file" | tail -1)
    local test_macc=$(grep -oP "test_macc: \K[\d.]+" "$log_file" | tail -1)
    local test_oa=$(grep -oP "test_oa: \K[\d.]+" "$log_file" | tail -1)

    # Return as JSON-like string
    echo "{\"best_epoch\":\"${best_epoch:-N/A}\",\"val_miou\":\"${val_miou:-N/A}\",\"val_macc\":\"${val_macc:-N/A}\",\"val_oa\":\"${val_oa:-N/A}\",\"test_miou\":\"${test_miou:-N/A}\",\"test_macc\":\"${test_macc:-N/A}\",\"test_oa\":\"${test_oa:-N/A}\"}"
}

# Function to send webhook notification
send_webhook() {
    local webhook_url="$1"
    local config_file="$2"
    local metrics="$3"
    local status="$4"
    local duration="$5"

    if [[ -z "$webhook_url" ]]; then
        echo -e "${YELLOW}No webhook URL provided. Skipping notification.${NC}"
        return
    fi

    local hostname=$(hostname)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Parse metrics
    local best_epoch=$(echo "$metrics" | grep -oP '"best_epoch":"\K[^"]+')
    local val_miou=$(echo "$metrics" | grep -oP '"val_miou":"\K[^"]+')
    local val_macc=$(echo "$metrics" | grep -oP '"val_macc":"\K[^"]+')
    local val_oa=$(echo "$metrics" | grep -oP '"val_oa":"\K[^"]+')
    local test_miou=$(echo "$metrics" | grep -oP '"test_miou":"\K[^"]+')
    local test_macc=$(echo "$metrics" | grep -oP '"test_macc":"\K[^"]+')
    local test_oa=$(echo "$metrics" | grep -oP '"test_oa":"\K[^"]+')

    # Create message
    local message="🎯 Training ${status}!

📊 Summary:
━━━━━━━━━━━━━━━━━━━━━━
⚙️  Config: ${config_file}
🖥️  Host: ${hostname}
⏱️  Duration: ${duration}
📅  Completed: ${timestamp}

📈 Validation Metrics:
  • Best Epoch: ${best_epoch}
  • mIoU: ${val_miou}
  • mAcc: ${val_macc}
  • OA: ${val_oa}

🎯 Test Metrics:
  • mIoU: ${test_miou}
  • mAcc: ${test_macc}
  • OA: ${test_oa}
━━━━━━━━━━━━━━━━━━━━━━"

    # Send to webhook (supports Google Chat, Discord, Slack, or generic webhooks)
    if [[ "$webhook_url" == *"chat.googleapis.com"* ]]; then
        # Google Chat webhook format with card
        local status_emoji="✅"
        local status_color="#34A853"  # Green
        if [[ "$STATUS" == "FAILED" ]]; then
            status_emoji="❌"
            status_color="#EA4335"  # Red
        fi

        local google_chat_payload=$(cat <<EOF
{
  "cards": [{
    "header": {
      "title": "${status_emoji} Training ${status}",
      "subtitle": "Point Cloud Model Training Report",
      "imageUrl": "https://www.gstatic.com/images/branding/product/2x/hh_48dp.png"
    },
    "sections": [{
      "widgets": [{
        "keyValue": {
          "topLabel": "Configuration",
          "content": "${config_file}",
          "contentMultiline": true,
          "icon": "DESCRIPTION"
        }
      }, {
        "keyValue": {
          "topLabel": "Host",
          "content": "${hostname}",
          "icon": "PERSON"
        }
      }, {
        "keyValue": {
          "topLabel": "Duration",
          "content": "${duration}",
          "icon": "CLOCK"
        }
      }, {
        "keyValue": {
          "topLabel": "Completed",
          "content": "${timestamp}",
          "icon": "EVENT_SEAT"
        }
      }]
    }, {
      "header": "📈 Validation Metrics",
      "widgets": [{
        "keyValue": {
          "topLabel": "Best Epoch",
          "content": "${best_epoch}"
        }
      }, {
        "keyValue": {
          "topLabel": "mIoU",
          "content": "${val_miou}",
          "icon": "STAR"
        }
      }, {
        "keyValue": {
          "topLabel": "mAcc",
          "content": "${val_macc}",
          "icon": "STAR"
        }
      }, {
        "keyValue": {
          "topLabel": "Overall Accuracy (OA)",
          "content": "${val_oa}",
          "icon": "STAR"
        }
      }]
    }, {
      "header": "🎯 Test Metrics",
      "widgets": [{
        "keyValue": {
          "topLabel": "mIoU",
          "content": "${test_miou}",
          "icon": "BOOKMARK"
        }
      }, {
        "keyValue": {
          "topLabel": "mAcc",
          "content": "${test_macc}",
          "icon": "BOOKMARK"
        }
      }, {
        "keyValue": {
          "topLabel": "Overall Accuracy (OA)",
          "content": "${test_oa}",
          "icon": "BOOKMARK"
        }
      }]
    }]
  }]
}
EOF
        )

        curl -H "Content-Type: application/json" \
             -X POST \
             -d "$google_chat_payload" \
             "$webhook_url" 2>/dev/null

    elif [[ "$webhook_url" == *"discord.com"* ]]; then
        # Discord webhook format
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\":\"${message}\"}" \
             "$webhook_url" 2>/dev/null
    elif [[ "$webhook_url" == *"slack.com"* ]]; then
        # Slack webhook format
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"text\":\"${message}\"}" \
             "$webhook_url" 2>/dev/null
    else
        # Generic webhook (JSON format)
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "{\"message\":\"${message}\",\"status\":\"${status}\",\"metrics\":${metrics},\"config\":\"${config_file}\",\"timestamp\":\"${timestamp}\"}" \
             "$webhook_url" 2>/dev/null
    fi

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Webhook notification sent successfully!${NC}"
    else
        echo -e "${RED}Failed to send webhook notification.${NC}"
    fi
}

# Function to cleanup RunPod instance
cleanup_runpod() {
    local pod_id="$1"

    if [[ -z "$pod_id" ]]; then
        echo -e "${YELLOW}No RunPod ID provided. Checking environment...${NC}"
        pod_id="${RUNPOD_POD_ID}"
    fi

    if [[ -z "$pod_id" ]]; then
        echo -e "${YELLOW}Running locally or no RUNPOD_POD_ID found. Skipping cleanup.${NC}"
        return
    fi

    echo -e "${CYAN}Initiating RunPod cleanup for pod: ${pod_id}${NC}"

    if command -v runpodctl &> /dev/null; then
        echo -e "${YELLOW}Removing RunPod instance...${NC}"
        runpodctl remove pod "$pod_id"

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}RunPod instance removed successfully!${NC}"
        else
            echo -e "${RED}Failed to remove RunPod instance.${NC}"
        fi
    else
        echo -e "${RED}runpodctl not found. Please install it first.${NC}"
        echo -e "${YELLOW}Install with: pip install runpodctl${NC}"
    fi
}

# Main execution
main() {
    # Print fancy banner
    print_banner

    # List and get config selection
    list_configs
    CONFIG_FILE=$(get_config_choice)

    echo -e "\n${GREEN}Selected config: ${BOLD}${CONFIG_FILE}${NC}\n"

    # Check for webhook URL from environment, otherwise ask
    if [[ -z "$WEBHOOK_URL" ]]; then
        echo -e "${BOLD}Enter webhook URL for notifications (press Enter to skip):${NC}"
        read -r WEBHOOK_URL
    else
        echo -e "${GREEN}Using webhook URL from environment${NC}"
    fi

    # Check for RunPod ID from environment, otherwise ask
    if [[ -z "$RUNPOD_POD_ID" ]]; then
        echo -e "${BOLD}Enter RunPod Pod ID for auto-cleanup (press Enter to skip):${NC}"
        read -r POD_ID
    else
        echo -e "${GREEN}Detected RunPod environment (Pod ID: ${RUNPOD_POD_ID})${NC}"
        POD_ID="$RUNPOD_POD_ID"
    fi

    # Confirmation
    echo -e "\n${YELLOW}${BOLD}Ready to start training!${NC}"
    echo -e "${CYAN}Config: ${CONFIG_FILE}${NC}"
    echo -e "${CYAN}Webhook: ${WEBHOOK_URL:-Not configured}${NC}"
    echo -e "${CYAN}RunPod ID: ${POD_ID:-${RUNPOD_POD_ID:-Not configured}}${NC}\n"

    echo -e "${BOLD}Press Enter to continue or Ctrl+C to cancel...${NC}"
    read

    # Start training
    echo -e "\n${GREEN}${BOLD}Starting training...${NC}\n"
    START_TIME=$(date +%s)

    # Create log directory
    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

    # Run training with logging
    uv run python examples/segmentation/main_pyg.py --cfg "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}

    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_FORMATTED=$(printf '%02dh:%02dm:%02ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

    echo -e "\n${CYAN}${BOLD}Training completed!${NC}"
    echo -e "${CYAN}Duration: ${DURATION_FORMATTED}${NC}\n"

    # Extract metrics
    echo -e "${YELLOW}Extracting training metrics...${NC}"
    METRICS=$(extract_metrics "$LOG_FILE")

    # Determine status
    if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
        STATUS="COMPLETED"
        echo -e "\n${GREEN}${BOLD}Training Summary:${NC}"
    else
        STATUS="FAILED"
        echo -e "\n${RED}${BOLD}Training Failed!${NC}"
    fi

    # Print metrics
    echo "$METRICS" | uv run python -c "import sys, json; data=json.loads(sys.stdin.read()); print('\n'.join([f'{k}: {v}' for k,v in data.items()]))"

    # Send webhook notification
    if [[ -n "$WEBHOOK_URL" ]]; then
        echo -e "\n${CYAN}Sending webhook notification...${NC}"
        send_webhook "$WEBHOOK_URL" "$CONFIG_FILE" "$METRICS" "$STATUS" "$DURATION_FORMATTED"
    fi

    # Cleanup RunPod instance
    if [[ "$STATUS" == "COMPLETED" ]]; then
        echo -e "\n${CYAN}${BOLD}Starting cleanup process...${NC}"
        sleep 3
        cleanup_runpod "${POD_ID:-${RUNPOD_POD_ID}}"
    else
        echo -e "\n${YELLOW}Skipping RunPod cleanup due to training failure.${NC}"
        echo -e "${YELLOW}Instance will remain running for debugging.${NC}"
    fi

    echo -e "\n${GREEN}${BOLD}All done!${NC}\n"
}

# Run main function
main

