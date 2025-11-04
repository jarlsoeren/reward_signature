from feature_extraction import extract_features

tasks = ['gonogo', 'hcp', 'twostep', 'mid', 'risksensitive'] #posner
out_types = ["z_score", "stat"]
out_dirs = ["/mnt/scratch/projects/rewardMap/reward_signature/lss_z_maps/", "/mnt/scratch/projects/rewardMap/reward_signature/lss_t_maps"]

task =  tasks[4]
out_dir = out_dirs[0]
out_type = out_types[0]


extract_features(task, out_type, out_dir)