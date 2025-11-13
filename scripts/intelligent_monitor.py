#!/usr/bin/env python3
"""
Intelligent Training Monitor
Automatically monitors training and stops if critical issues are detected
"""

import os
import sys
import time
import re
import signal
import psutil
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import argparse

class TrainingMonitor:
    """Monitor training process and detect anomalies"""

    def __init__(self, log_file: str, pid: int, check_interval: int = 30):
        self.log_file = Path(log_file)
        self.pid = pid
        self.check_interval = check_interval

        # Thresholds for stopping
        self.max_error_rate = 0.80  # Stop if >80% of samples are erroring
        self.min_progress_time = 300  # Stop if no progress for 5 minutes
        self.max_consecutive_errors = 50  # Stop if 50 consecutive errors

        # Deep check settings (every 30 minutes)
        self.deep_check_interval = 1800  # 30 minutes in seconds
        self.last_deep_check = time.time()
        self.deep_check_history = []  # Track progress over time

        # Tracking
        self.last_position = 0
        self.last_progress_time = time.time()
        self.error_counts = defaultdict(int)
        self.warning_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=100)  # Track last 100 operations
        self.consecutive_errors = 0
        self.total_samples_processed = 0
        self.successful_samples = 0
        self.start_time = time.time()

    def read_new_logs(self) -> str:
        """Read new log lines since last check"""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()
                return new_content
        except Exception as e:
            print(f"Error reading log: {e}")
            return ""

    def analyze_logs(self, content: str) -> dict:
        """Analyze log content for errors and progress"""
        analysis = {
            'errors': [],
            'warnings': [],
            'progress': None,
            'has_progress': False
        }

        for line in content.split('\n'):
            # Track progress
            if 'Testing' in line or 'Training' in line:
                match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)', line)
                if match:
                    percent, current, total = match.groups()
                    analysis['progress'] = {
                        'percent': int(percent),
                        'current': int(current),
                        'total': int(total)
                    }
                    analysis['has_progress'] = True
                    self.last_progress_time = time.time()
                    self.total_samples_processed = int(current)

            # Track errors
            if 'ERROR' in line:
                # Extract error type
                if 'Slither analysis failed' in line:
                    self.error_counts['slither'] += 1
                    self.recent_errors.append('slither')
                    self.consecutive_errors += 1
                elif 'Mythril analysis failed' in line:
                    self.error_counts['mythril'] += 1
                    self.recent_errors.append('mythril')
                    self.consecutive_errors += 1
                elif 'Training error' in line or 'encoder error' in line:
                    self.error_counts['training'] += 1
                    self.recent_errors.append('training')
                    self.consecutive_errors += 1
                else:
                    self.error_counts['other'] += 1
                    self.recent_errors.append('other')
                    self.consecutive_errors += 1

                analysis['errors'].append(line.strip())

            # Track warnings
            if 'WARNING' in line:
                if 'Slither' in line:
                    self.warning_counts['slither'] += 1
                elif 'Mythril' in line:
                    self.warning_counts['mythril'] += 1
                analysis['warnings'].append(line.strip())

            # Reset consecutive errors on success
            if 'successful' in line.lower() or 'completed' in line.lower():
                self.consecutive_errors = 0

        return analysis

    def check_error_rate(self) -> tuple:
        """Check if error rate is too high"""
        if len(self.recent_errors) < 20:
            return False, 0.0

        error_rate = len(self.recent_errors) / 100.0
        if error_rate > self.max_error_rate:
            return True, error_rate

        return False, error_rate

    def check_stalled(self) -> tuple:
        """Check if training has stalled"""
        time_since_progress = time.time() - self.last_progress_time
        if time_since_progress > self.min_progress_time:
            return True, time_since_progress
        return False, time_since_progress

    def check_consecutive_errors(self) -> bool:
        """Check if too many consecutive errors"""
        return self.consecutive_errors > self.max_consecutive_errors

    def deep_check(self) -> tuple:
        """
        Deep check every 30 minutes to assess if training is still worthwhile.
        Returns (should_stop, reason, detailed_report)
        """
        elapsed_time = time.time() - self.start_time
        elapsed_hours = elapsed_time / 3600

        # Calculate overall statistics
        total_errors = sum(self.error_counts.values())
        total_warnings = sum(self.warning_counts.values())

        # Calculate success rate
        if self.total_samples_processed > 0:
            # Each sample goes through multiple tools (Slither, Mythril)
            # Estimate total operations
            estimated_operations = self.total_samples_processed * 4  # rough estimate
            error_rate = total_errors / max(estimated_operations, 1)
            success_rate = 1.0 - error_rate
        else:
            error_rate = 0.0
            success_rate = 1.0

        # Estimate time remaining
        if self.total_samples_processed > 0:
            samples_per_hour = self.total_samples_processed / max(elapsed_hours, 0.01)
            # Assuming typical dataset size
            estimated_total = 10000  # adjust based on your dataset
            remaining_samples = max(estimated_total - self.total_samples_processed, 0)
            estimated_hours_remaining = remaining_samples / max(samples_per_hour, 1)
        else:
            estimated_hours_remaining = 0

        # Store in history
        checkpoint = {
            'timestamp': datetime.now(),
            'elapsed_hours': elapsed_hours,
            'samples_processed': self.total_samples_processed,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'success_rate': success_rate,
            'slither_errors': self.error_counts['slither'],
            'mythril_errors': self.error_counts['mythril'],
            'training_errors': self.error_counts['training']
        }
        self.deep_check_history.append(checkpoint)

        # Build detailed report
        report = f"""
{'='*80}
DEEP CHECK - Training Health Assessment
{'='*80}
Time Elapsed: {elapsed_hours:.1f} hours
Estimated Time Remaining: {estimated_hours_remaining:.1f} hours
Samples Processed: {self.total_samples_processed}

Error Statistics:
  Total Errors: {total_errors}
  - Slither failures: {self.error_counts['slither']}
  - Mythril failures: {self.error_counts['mythril']}
  - Training errors: {self.error_counts['training']}
  - Other errors: {self.error_counts['other']}

Success Rate: {success_rate*100:.1f}%
Error Rate: {error_rate*100:.1f}%

Recent Performance (last 100 samples):
  Recent Error Rate: {len(self.recent_errors):.0f}%
  Consecutive Errors: {self.consecutive_errors}

"""

        # Decision logic: Is training becoming useless?
        should_stop = False
        reason = ""

        # Criterion 1: If error rate is very high and we have enough data
        if error_rate > 0.85 and self.total_samples_processed > 50:
            should_stop = True
            reason = (f"CRITICAL: Training is {error_rate*100:.0f}% failing. "
                     f"Model would be trained on mostly empty/invalid data. "
                     f"Stopping to prevent wasting {estimated_hours_remaining:.1f} more hours of GPU time.")

        # Criterion 2: If Slither is failing almost everywhere
        elif (self.error_counts['slither'] > self.total_samples_processed * 0.8
              and self.total_samples_processed > 100):
            should_stop = True
            reason = (f"CRITICAL: Slither analysis failing on {self.error_counts['slither']}/{self.total_samples_processed} samples. "
                     f"Static encoder will have no useful features to learn from. "
                     f"Training has become useless.")

        # Criterion 3: If error rate is increasing over time
        elif len(self.deep_check_history) >= 3:
            recent_error_rates = [h['error_rate'] for h in self.deep_check_history[-3:]]
            if all(recent_error_rates[i] > recent_error_rates[i-1] for i in range(1, len(recent_error_rates))):
                if recent_error_rates[-1] > 0.7:
                    should_stop = True
                    reason = (f"CRITICAL: Error rate is increasing over time and now at {recent_error_rates[-1]*100:.0f}%. "
                             f"Training quality is deteriorating. Stopping before complete failure.")

        # Criterion 4: If training errors (not just analysis failures) are too high
        elif self.error_counts['training'] > 50:
            should_stop = True
            reason = (f"CRITICAL: {self.error_counts['training']} training errors detected. "
                     f"The model itself is failing to train properly. "
                     f"This indicates a fundamental problem.")

        report += f"\nDecision: {'STOP TRAINING' if should_stop else 'CONTINUE'}\n"
        if should_stop:
            report += f"Reason: {reason}\n"
        else:
            report += f"Training is progressing acceptably. Will check again in 30 minutes.\n"

        report += f"{'='*80}\n"

        return should_stop, reason, report

    def is_process_alive(self) -> bool:
        """Check if training process is still running"""
        try:
            process = psutil.Process(self.pid)
            return process.is_running()
        except:
            return False

    def stop_training(self, reason: str):
        """Stop the training process"""
        print(f"\n{'='*80}")
        print(f"STOPPING TRAINING")
        print(f"{'='*80}")
        print(f"Reason: {reason}")
        print(f"\nError Summary:")
        for error_type, count in self.error_counts.items():
            print(f"  {error_type}: {count} errors")
        print(f"\nWarning Summary:")
        for warning_type, count in self.warning_counts.items():
            print(f"  {warning_type}: {count} warnings")
        print(f"\n{'='*80}")

        try:
            os.kill(self.pid, signal.SIGTERM)
            print(f"Sent SIGTERM to process {self.pid}")
            time.sleep(5)

            # Force kill if still alive
            if self.is_process_alive():
                os.kill(self.pid, signal.SIGKILL)
                print(f"Sent SIGKILL to process {self.pid}")
        except Exception as e:
            print(f"Error stopping process: {e}")

    def print_status(self, analysis: dict):
        """Print current status"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if analysis['progress']:
            p = analysis['progress']
            print(f"[{timestamp}] Progress: {p['current']}/{p['total']} ({p['percent']}%)")

        if len(self.recent_errors) > 0:
            error_rate = len(self.recent_errors) / 100.0
            print(f"[{timestamp}] Recent error rate: {error_rate:.1%} "
                  f"({len(self.recent_errors)}/100 samples)")

        print(f"[{timestamp}] Total errors: {sum(self.error_counts.values())}, "
              f"Consecutive: {self.consecutive_errors}")

    def monitor(self):
        """Main monitoring loop"""
        print(f"Starting intelligent monitoring of training process {self.pid}")
        print(f"Log file: {self.log_file}")
        print(f"Check interval: {self.check_interval}s")
        print(f"Deep check interval: {self.deep_check_interval}s (30 minutes)")
        print(f"\nMonitoring thresholds:")
        print(f"  Max error rate: {self.max_error_rate:.0%}")
        print(f"  Max consecutive errors: {self.max_consecutive_errors}")
        print(f"  Stall timeout: {self.min_progress_time}s")
        print(f"\n{'='*80}\n")

        try:
            while True:
                # Check if process is still alive
                if not self.is_process_alive():
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Training process {self.pid} has terminated.")
                    break

                # Read and analyze new logs
                new_content = self.read_new_logs()
                if new_content:
                    analysis = self.analyze_logs(new_content)

                    # Print status
                    if analysis['has_progress']:
                        self.print_status(analysis)

                    # Check for critical issues (quick checks)
                    should_stop = False
                    stop_reason = ""

                    # Check 1: High error rate
                    is_high_error, error_rate = self.check_error_rate()
                    if is_high_error:
                        should_stop = True
                        stop_reason = f"High error rate: {error_rate:.1%} of recent samples failing"

                    # Check 2: Too many consecutive errors
                    if self.check_consecutive_errors():
                        should_stop = True
                        stop_reason = f"Too many consecutive errors: {self.consecutive_errors}"

                    # Check 3: Training stalled
                    is_stalled, stall_time = self.check_stalled()
                    if is_stalled:
                        should_stop = True
                        stop_reason = f"Training stalled: no progress for {stall_time:.0f}s"

                    # Stop if critical issue detected
                    if should_stop:
                        self.stop_training(stop_reason)
                        break

                # Deep check every 30 minutes
                time_since_deep_check = time.time() - self.last_deep_check
                if time_since_deep_check >= self.deep_check_interval:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Performing deep health check (30-minute checkpoint)...")

                    should_stop, reason, report = self.deep_check()
                    print(report)

                    if should_stop:
                        self.stop_training(reason)
                        break

                    self.last_deep_check = time.time()

                # Wait before next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nMonitoring error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Intelligent Training Monitor")
    parser.add_argument("--log-file", required=True, help="Path to training log file")
    parser.add_argument("--pid", type=int, required=True, help="Training process PID")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)

    try:
        psutil.Process(args.pid)
    except:
        print(f"Error: Process {args.pid} not found")
        sys.exit(1)

    monitor = TrainingMonitor(args.log_file, args.pid, args.interval)
    monitor.monitor()


if __name__ == "__main__":
    main()
