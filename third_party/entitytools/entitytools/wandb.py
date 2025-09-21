import wandb
import pprint

def tune_with_wandb(
    project,
    sweep_id,
    sweep_method,
    metric_name,
    metric_goal,
    training_func,
    fixed_args,
    tunable_args):

    parameters_dict = { name: {'values':values} for name,values in tunable_args.items() }   
    
    sweep_config = {
        "method": sweep_method,
        "metric": {
            "name": metric_name,
            "goal": metric_goal
        },
        "parameters": parameters_dict
    }
        
    pprint.pprint(sweep_config)
    
    def wandb_train(config=None):    
        with wandb.init(config=config):
            config = wandb.config

            args = { 'wandb_logger': wandb }
            for arg_name,value in fixed_args.items():
                args[arg_name] = value
            for arg_name in tunable_args:
                args[arg_name] = getattr(config, arg_name)
            
            training_func(**args)
            
    if not sweep_id:
        sweep_id = wandb.sweep(sweep_config, project=project)
    
    wandb.agent(sweep_id, wandb_train, project=project)