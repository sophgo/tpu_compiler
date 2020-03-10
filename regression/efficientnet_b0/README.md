# efficientnet b0

efficientnet b0, download from onedrive link
deploy.txt goes here

- `https://iclinktw-my.sharepoint.com/:f:/g/personal/sam_zheng_wisecore_com_tw/EnP2WyYVYTxJhwtkagBwoAEBzABcg_KC_jWXgAjGegv_dg?e=0awjit`

## Performance

## Accuracy

- 20200310 (1000 images)

  | mode             | Top-1 (%) | Top-5 (%) |
  | ---              | ---       | ---       |
  | caffe original   | 60.900    | 84.600    |
  | fp32             | 59.600    | 84.000    |
  | int8 Per-layer   | -         | -         |
  | int8 Per-channel | -         | -         |
  | int8 Multiplier  | 21.600    | 41.300    |
  | fp16             | -         | -         |

- 20200309 (1000 images)

  | mode             | Top-1 (%) | Top-5 (%) |
  | ---              | ---       | ---       |
  | caffe original   | 57.900    | 81.900    |
  | fp32             | 58.400    | 83.000    |
  | int8 Per-layer   | -         | -         |
  | int8 Per-channel | 18.500    | 37.700    |
  | int8 Multiplier  | 19.700    | 37.400    |
  | fp16             | 60.100    | 82.400    |
