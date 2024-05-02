from algorithms.sac import SAC
from algorithms.sac_aug import SAC_AUG
from algorithms.drq import DrQ
from algorithms.drq_aug import DrQ_AUG
from algorithms.svea_c import SVEA_C
from algorithms.svea_c_aug import SVEA_C_AUG


algorithm = {
	'sac': SAC,
	'sac_aug':SAC_AUG,
	'drq': DrQ,
	'drq_aug': DrQ_AUG,
	'svea_c':SVEA_C,
	'svea_c_aug':SVEA_C_AUG,
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
