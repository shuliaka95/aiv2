# debug_component.py
import sys
sys.path.insert(0, '.')
from torchsig.datasets import TorchSigIterableDataset, DatasetMetadata

metadata = DatasetMetadata(
    num_iq_samples_dataset=1024,
    fft_size=1024,
    class_list=['ook', 'bpsk', 'qpsk'],
    sample_rate=10e6,
    kwargs={}
)

dataset = TorchSigIterableDataset(dataset_metadata=metadata, transforms=[])
signal = next(iter(dataset))

print("ДИАГНОСТИКА COMPONENT_SIGNALS")
print("=" * 60)

if hasattr(signal, 'component_signals'):
    print(f"Количество component_signals: {len(signal.component_signals)}")
    
    for i, comp in enumerate(signal.component_signals[:3]):  # Первые 3
        print(f"\nComponent {i}:")
        print(f"  Тип: {type(comp)}")
        
        if hasattr(comp, 'metadata') and comp.metadata is not None:
            print(f"  metadata: {comp.metadata}")
            print(f"  Тип metadata: {type(comp.metadata)}")
            
            # Покажем все атрибуты metadata
            meta_attrs = [attr for attr in dir(comp.metadata) if not attr.startswith('_')]
            print(f"  Атрибуты metadata: {meta_attrs[:15]}")
            
            # Покажем значения ключевых атрибутов
            for attr in ['class_idx', 'class_name', 'modulation', 'label', 'class_index']:
                if hasattr(comp.metadata, attr):
                    value = getattr(comp.metadata, attr)
                    print(f"    {attr}: {value} (тип: {type(value)})")
        else:
            print(f"  metadata: {comp.metadata}")
else:
    print("Signal не имеет component_signals")