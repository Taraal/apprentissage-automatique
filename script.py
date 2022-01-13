from jiant.proj.simple import runscript as run

import jiant.scripts.download_data.runscript as downloader
import jiant.proj.main.scripts.configurator as configurator
import jiant.utils.python.io as py_io
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching

import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import os

EXP_DIR = "td/try"

# Download the Data

tasks = ["wic", "wsc", "rte"]

#downloader.download_data(tasks, f"{EXP_DIR}/tasks")

#for task in tasks:
def train_jiant(config):
	for task in tasks:
		downloader.download_data([task], f"{EXP_DIR}/tasks")

# Set up the arguments for the Simple API
	args = run.RunConfiguration(
	   run_name="simple",
	   exp_dir=EXP_DIR,
	   data_dir=f"{EXP_DIR}/tasks",
	   hf_pretrained_model_name_or_path="roberta-base",
	   tasks=','.join(tasks), # TO FIX
		train_batch_size=config['batch_size'], # default = 16
		num_train_epochs=config["epochs"], # default = 3
		learning_rate=config["lr"], # default = 1e-5
	)

	# Run!
	run.run_simple(args)
	val_metrics = py_io.read_json(os.path.join(EXP_DIR, "runs", 'simple', "val_metrics.json"))
	tune.report(aggregated=val_metrics["aggregated"]) # Changer ici le nom et l'empaclement de la métrique

search_space = {
	"lr": tune.grid_search([1e-5, 5e-5, 5e-6]),  #tune.sample_from(lambda spec: 10**(-10 * np.random.rand())), # Changer ici les valeurs possibles pour LR
	"batch_size": tune.grid_search([4, 8, 16]),
	"epochs": tune.grid_search([2, 3, 4])
}


analysis = tune.run(train_jiant,
	metric="aggregated", # Changer ici le nom de la métrique
	mode="max",
	config=search_space,
	num_samples=1, # Changer ici le nb de trials 
	resources_per_trial={"gpu": 1, "cpu": 13})

dfs = analysis.trial_dataframes

print("#######")
print("BEST CONFIG : ")
print(analysis.best_config)
print("#######")

print()
print("######")
print("BEST RESULT : ")
print(analysis.best_result)
print("######")
