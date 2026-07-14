from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod, DeepFool, SaliencyMapMethod, BoundaryAttack
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import ZooAttack

def fgsm_attack(classifier,X_scaled_test, eps): 

    # Generate adversarial examples using FGSM
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=X_scaled_test.values)

    return X_test_adv

def pgd_attack(classifier, X_scaled_test, eps):
    attack = ProjectedGradientDescent(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=X_scaled_test.values)
    return X_test_adv

def bim_attack(classifier, X_scaled_test, eps):
    attack = BasicIterativeMethod(estimator=classifier, eps=eps)
    X_test_adv = attack.generate(x=X_scaled_test.values)
    return X_test_adv

"""
def deepfool_attack(classifier, X_scaled_test):
    attack = DeepFool(classifier=classifier)
    X_test_adv = attack.generate(x=X_scaled_test.values)
    return X_test_adv
"""
def jsma_attack(classifier, X_scaled_test, theta, gamma):
    attack = SaliencyMapMethod(classifier=classifier, theta=theta, gamma=gamma)
    X_test_adv = attack.generate(x=X_scaled_test)
    return X_test_adv

def cw_attack(classifier, X_scaled_test):
    attack = CarliniL2Method(classifier=classifier)
    
    # Generate adversarial examples
    X_test_adv = attack.generate(x=X_scaled_test.values)
    
    return X_test_adv


def zoo_attack(classifier, X_scaled_test):

    attack = ZooAttack(classifier=classifier,
    confidence=0.5,  # Increase confidence to make the attack stronger
    targeted=False,
    learning_rate=0.01,  # Adjust learning rate as needed
    max_iter=20,  # Increase maximum iterations for more potent attacks
    binary_search_steps=10,  # Adjust binary search steps if needed
    initial_const=1.0,  # Increase initial constant
    abort_early=False,  # Disable early stopping
    use_resize=False,
    use_importance=False,
    nb_parallel=20,
    batch_size=1,
    variable_h=0.1
)
    # Generate adversarial examples
    X_test_adv = attack.generate(X_scaled_test)

    return X_test_adv

# add to imports at the top
from art.attacks.evasion import HopSkipJump, SquareAttack

def hsj_attack(classifier, X_scaled_test, max_iter=25, max_eval=10000, init_eval=100):
    """
    HopSkipJump (decision-based black-box).
    Works on tabular as long as clip_values are set on the classifier.
    """
    attack = HopSkipJump(
        classifier=classifier,
        targeted=False,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=True
    )
    X_test_adv = attack.generate(x=X_scaled_test.values if hasattr(X_scaled_test, "values") else X_scaled_test)
    return X_test_adv

def square_attack(classifier, X_scaled_test, eps=0.2, max_iter=5000):
    """
    Square Attack (score-based black-box, L_inf by default).
    """
    attack = SquareAttack(
        estimator=classifier,
        eps=eps,
        max_iter=max_iter,
        p_init=0.1,   # starting fraction of perturbed features
        nb_restarts=1
    )
    X_test_adv = attack.generate(x=X_scaled_test.values if hasattr(X_scaled_test, "values") else X_scaled_test)
    return X_test_adv

def boundary_attack(classifier, X_scaled_test,
                    targeted=False, max_iter=1000, epsilon=0.01,
                    step_adapt=0.667, init_size=100, batch_size=64):
    """
    BoundaryAttack (decision-based black-box).
    """
    attack = BoundaryAttack(
        estimator=classifier,
        targeted=targeted,
        max_iter=max_iter,
        epsilon=epsilon,
        step_adapt=step_adapt,
        batch_size=batch_size,
        init_size=init_size
    )
    X = X_scaled_test.values if hasattr(X_scaled_test, "values") else X_scaled_test
    return attack.generate(x=X)