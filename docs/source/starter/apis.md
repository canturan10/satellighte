
# APIs

## 1- Get Available Models

```python
import satellighte as sat
sat.available_models()
# ['efficientnet_b0_eurosat', 'mobilenetv2_default_eurosat']
```

## 2- Get Available Versions for a Spesific Model

```python
import satellighte as sat
model_name = 'efficientnet_b0_eurosat'
sat.get_model_versions(model_name)
# ['0']
```

## 3- Get Latest Version for a Spesific Model

```python
import satellighte as sat
model_name = 'efficientnet_b0_eurosat'
sat.get_model_latest_version(model_name)
# '0'
```

## 4- Get Pretrained Model

```python
import satellighte as sat
model_name = 'efficientnet_b0_eurosat'
model = sat.Classifier.from_pretrained(model_name, version=None) # if version none is given than latest version will be used.
# model: pl.LightningModule
```

## 5- Get Model with Random Weight Initialization

```python
import satellighte as sat
arch = 'efficientnet'
config = 'b0'
model = sat.Classifier.build(arch, config)
# model: pl.LightningModule
```

## 6- Get Pretrained Arch Model

```python
import satellighte as sat
model_name = 'efficientnet_b0_eurosat'
model = sat.Classifier.from_pretrained_arch(model_name, version=None) # if version none is given than latest version will be used.
# model: torch.nn.Module
```

## 7- Get Arch Model with Random Weight Initialization

```python
import satellighte as sat
arch = 'efficientnet'
config = 'b0'
model = sat.Classifier.build_arch(arch, config)
# model: torch.nn.Module
```