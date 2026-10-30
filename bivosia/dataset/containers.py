"""Open structured result containers through their registered storage format."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.formats import format_handler


def open_container(path, *, format=None):
    """Open a container; dataset-specific loaders interpret its schema."""
    return format_handler('containers', path, format=format)(path)


def container_output(path, *, overwrite=True, format=None):
    """Publish a closed container on success and remove failed partial output."""
    return format_handler('containers', path, format=format, operation='writer')(path, overwrite=overwrite)
