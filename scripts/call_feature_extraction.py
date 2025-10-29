from feature_extraction import extract_features

tasks = ['gonogo', 'hcp', 'twostep', 'mid', 'risksensitive'] #posner
out_types = ["z_score", "stat"]
out_dirs = ["../lss_z_maps", "../lss_t_maps"]

task =  tasks[4]
out_dir = out_dirs[1]
out_type = out_types[1]


extract_features(task, out_type, out_dir)