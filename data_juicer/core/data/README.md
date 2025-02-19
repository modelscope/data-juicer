# Dataset Configuration Guide

This guide provides an overview of how to configure datasets using YAML files in the Data-Juicer framework. The configurations allow you to specify local and remote datasets, config validation rules, and data validation rules.

## Configuration Files

### Local JSON Dataset

The `local_json.yaml` configuration file is used to specify datasets stored locally in JSON format.

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.json
      format: json
    - type: local
      path: path/to/your/local/dataset.json
      format: json
```

### Remote JSON Dataset

The `remote_json.yaml` configuration file is used to specify datasets stored remotely in JSON format.


```yaml
remote_json:
  path: https://example.com/dataset.json
  format: json
  schema:
```

### Ray Dataset

The `ray.yaml` configuration file is used to specify datasets stored in Ray format.

```yaml
ray:
```

### Nested Dataset

The `nested.yaml` configuration file is used to specify datasets stored in nested format.

```yaml
nested:
```

### Data Validator

The `validator.yaml` configuration file is used to specify the data validator to use.

```yaml
validator:


```


