# Date Truncate Function

The date\_trunc function supports the given below units. The examples in the table use timestamp 2000-08-05T09:05:10 as an input.

|   unit  | Truncated example value |
| :-----: | :---------------------: |
|  second |   2000-08-05T09:05:10   |
|  minute |   2000-08-05T09:05:00   |
|   hour  |   2000-08-05T09:00:00   |
|   day   |   2000-08-05T00:00:00   |
|   week  |   2000-07-31T00:00:00   |
|  month  |   2000-08-01T00:00:00   |
| quarter |   2000-07-01T00:00:00   |
|   year  |   2000-01-01T00:00:00   |

####

#### <mark style="color:purple;">DATE\_TRUNC(</mark>  unit,  \<datetime\_expr>  <mark style="color:purple;">)</mark>

Returns the input expression truncated to a given `unit` (refer above table).

```sql
 > select date_trunc('year' , '2022-03-23');
 '2022-01-01'
```

