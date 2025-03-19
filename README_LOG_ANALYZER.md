# System Log Analyzer with Edge TPU

This project uses the Coral USB Accelerator to detect anomalies in system logs.

## Setup Instructions

1. Install dependencies:
   ```
   poetry add scikit-learn pandas
   ```

2. Collect training data:
   ```
   python -m project_keyword_spotter.collect_training_logs
   ```

3. Train and compile the model (instructions in the training notebook)

4. Run the analyzer:
   ```
   ./run_log_analyzer.sh
   ```

## Project Structure

- `project_keyword_spotter/log_analyzer.py`: Main implementation file
- `project_keyword_spotter/collect_training_logs.py`: Script to gather training data
- `models/log_analyzer/`: Contains the TFLite model and vectorizer
- `config/log_analyzer/`: Configuration files
- `data/log_analyzer/`: Training and test data

## Automation

To run the analyzer automatically, add this to crontab:

```
*/30 * * * * cd /home/aygp-dr/projects/edgetpu-ml/opt/project-keyword-spotter && ./run_log_analyzer.sh >> /home/aygp-dr/log_analyzer.log 2>&1
```

This will run the analyzer every 30 minutes.
