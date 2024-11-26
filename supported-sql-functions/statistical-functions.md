---
description: Uncategorized additional functions supported by e6data
---

# Statistical Functions

#### <mark style="color:purple;">STDDEV(</mark>  expression  <mark style="color:purple;">)</mark>

Returns the sample standard deviation, within the input group of values.

```sql
select stddev(colB) from table1
```

#### <mark style="color:purple;">STDDEV\_POP(</mark>  expression  <mark style="color:purple;">)</mark>

Returns the population standard deviation, which is the square root of the population variance, from the input group of values.

```sql
select stddev_pop(colB) from table1
```

#### <mark style="color:purple;">PERCENTILE\_CONT(</mark> \<percentile> ) WITHIN GROUP (ORDER BY \<expression> \[ ASC | DESC ] <mark style="color:purple;">)</mark>

Returns the linear interpolated value that would fall into the given **\<percentile>** value with respect to the specified sort direction.

```sql
select percentile_cont(0.5) within group1 (order by colA asc) from table1
```



