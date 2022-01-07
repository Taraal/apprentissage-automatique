from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import jiant.utils.python.io as py_io
import os

EXP_DIR = "td/try"

# Download the Data

tasks = ["rte", "wic", "wsc", "mrpc"]

#downloader.download_data(tasks, f"{EXP_DIR}/tasks")

task = "mrpc"
#for task in tasks:
def train_jiant(config):
	downloader.download_data(["mrpc"], f"{EXP_DIR}/tasks")

# Set up the arguments for the Simple API
	args = run.RunConfiguration(
	   run_name="simple",
	   exp_dir=EXP_DIR,
	   data_dir=f"{EXP_DIR}/tasks",
	   hf_pretrained_model_name_or_path="roberta-base",
	   tasks=task,
	   train_batch_size=16,
		num_train_epochs=3,
		learning_rate=config["lr"]
	)

	# Run!
	run.run_simple(args)
	
	val_metrics = py_io.read_json(os.path.join(EXP_DIR, "runs", 'simple', "val_metrics.json"))
	tune.report(major=val_metrics[task]["metrics"]["major"])

search_space = {
	"lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand()))
}


analysis = tune.run(train_jiant,
	metric="major",
	mode="max",
	config=search_space, num_samples=1,
	resources_per_trial={"gpu": 1, "cpu": 13})

dfs = analysis.trial_dataframes
# print([d.mean_accuracy for d in dfs.values()])
#print(dfs)
print("#######")
print("BEST CONFIG : ")
print(analysis.best_config)
print("#######")
