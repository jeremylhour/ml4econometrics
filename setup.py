from setuptools import setup

if __name__ == "__main__":
    setup(
        name="ml4econometrics",
        description="Companion repo for Machine Learning for Econometrics textbook",
        author="Jérémy L'Hour",
        author_email="jeremy.l.hour@ensae.fr",
        license="MIT License",
        packages=setuptools.find_packages(),
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires=">=3.6",
    )