# Cluster Output Capture

## kubectl describe <deployment>

```
# paste output of: kubectl -n auto-mlops describe deploy/model-api
```

## kubectl describe <pod>

```
# paste output of: kubectl -n auto-mlops describe pod -l app=model-api
```

## kubectl describe <ingress>

```
# paste output of: kubectl -n auto-mlops describe ingress auto-mlops
```

## kubectl top pod

```
# paste output of: kubectl -n auto-mlops top pod
```

## kubectl top node

```
# paste output of: kubectl top node
```

## kubectl get all -A -o yaml

```
# paste output of: kubectl get all -A -o yaml
```
