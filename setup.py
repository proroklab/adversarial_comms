import os
import setuptools

package_dir = os.path.dirname(os.path.realpath(__file__))

with open(package_dir + "/README.md", "r") as fh:
    long_description = fh.read()

requirements_dir = package_dir + '/requirements.txt'
install_requires = []
with open(requirements_dir) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="adversarial-comms",
    version="1.0.0",
    author="Jan Blumenkamp",
    author_email="jan.blumenkamp@gmx.de",
    description="Package accompanying the paper 'The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/proroklab/adversarial_comms",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    entry_points = {
        'console_scripts': [
            'train_policy=adversarial_comms.train_policy:start_experiment',
            'continue_policy=adversarial_comms.train_policy:continue_experiment',
            'evaluate_coop=adversarial_comms.evaluate:eval_nocomm_coop',
            'evaluate_adv=adversarial_comms.evaluate:eval_nocomm_adv',
            'evaluate_random=adversarial_comms.evaluate:eval_random',
            'evaluate_plot=adversarial_comms.evaluate:plot',
        ],
    },
    python_requires='>=3.7',
)
