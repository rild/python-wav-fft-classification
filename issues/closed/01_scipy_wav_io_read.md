## Waht
Some wave file would not be read.

cause following error 

```
raise ValueError("Incomplete wav chunk")
```

It seems to be neccessary to get wave format knowledge..

## Approach
change import module: scipy.io.wave to wave

- LGTM
