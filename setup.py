import ez_setup
#ez_setup.use_setuptools()
from setuptools import setup, find_packages
setup(
    name = "django-cookie",
    version = "0.1",
    packages = find_packages(),
    author = "Paridhi",
    author_email = "visit.paridhi@gmail.com",
    description = "A package to help automate creation of testing in Django",
    url = "http://code.google.com/p/django-testmaker/",
    include_package_data = True,
    scripts=['manage.py']
)
