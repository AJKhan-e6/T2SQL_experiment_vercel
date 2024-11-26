---
description: >-
  Aggregate functions operate on multiple sets of values and return a single
  value.
---

# Aggregate Functions

* [AVG](aggregate-functions.md#avg-less-than-expr-greater-than)
* [COUNT](aggregate-functions.md#count-less-than-expr-greater-than)
* [DISTINCT](aggregate-functions.md#distinct-less-than-expr-greater-than)
* [MAX](aggregate-functions.md#max-less-than-expr-greater-than)
* [MIN](aggregate-functions.md#min-less-than-expr-greater-than)
* [SUM](aggregate-functions.md#sum-less-than-expr-greater-than)
* [APPROX\_QUANTILES](aggregate-functions.md#approx\_quantiles-less-than-expr-greater-than-number)

#### <mark style="color:purple;">AVG(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the arithmetic mean of all input values.

```sql
select avg(colA) from table1;
```

#### <mark style="color:purple;">COUNT(</mark>  \<expr>  <mark style="color:purple;">)</mark>&#x20;

Returns the number of rows in an input value.

```sql
select count(colA) from table1;

select count(*), colA from table1 group by colA;
```

#### <mark style="color:purple;">DISTINCT(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the number of distinct values in a given column.

```sql
select distinct(colA) from table1;
```

#### <mark style="color:purple;">MAX(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the maximum value of all input values.

```sql
select max(colA) from table1;
```

#### <mark style="color:purple;">MIN(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the minimum value of all input values.

```sql
select min(colA) from table1;
```

#### <mark style="color:purple;">SUM(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the arithmetic sum of all input values.

```sql
select sum(colA) from table1;
```

#### <mark style="color:purple;">APPROX\_QUANTILES(</mark>  \<expr>,   number  <mark style="color:purple;">)</mark>

```sql
select approx_quantiles(colA, 4)
```

Returns the approximate boundaries for a group of `<expr>`values, where `number` represents the number of quantiles to create. This function returns an array of `number` + 1 element, where the first element is the approximate minimum and the last element is the approximate maximum.

#### <mark style="color:purple;">GREATEST(</mark>  \<expr1>,  \<expr2>,  \<expr3>......\<exprN>  <mark style="color:purple;">)</mark>

Returns the largest value among the given values.

```sql
select greatest(100, 12, 23, 1999, 2)
1999
```

#### <mark style="color:purple;">LEAST(</mark>  \<expr1>,  \<expr2>,  \<expr3>......\<exprN>  )

Returns the smallest value among the given values.

```sql
select least(-12, 0, 10000, -527)
-527 
```

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
