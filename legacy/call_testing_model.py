from testing_model import run_model

tasks = ['gonogo', 'hcp', 'mid', 'risksensitive', 'twostep'] #posner
n_subs = [15]
maps_dir = '../lss_maps'

for i in range(len(tasks)):
    train_tasks = tasks[:i] + tasks[i+1:]
    test_tasks = [tasks[i]]

    for n in n_subs:
        print(f"Train model on {train_tasks}\nTest model on {test_tasks}\nnumber of subs: {n}")
        run_model(train_tasks, test_tasks, n, maps_dir)
        print("\n\n")
