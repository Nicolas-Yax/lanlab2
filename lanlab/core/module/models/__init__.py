from lanlab.core.module.models.openai_models import get_openai_model_classes
from lanlab.core.module.models.model import set_number_workers
from lanlab.core.module.models.hf_models import get_hf_model_classes

for cls in get_openai_model_classes():
    globals()[cls.__name__] = cls

for cls in get_hf_model_classes():
    globals()[cls.__name__] = cls