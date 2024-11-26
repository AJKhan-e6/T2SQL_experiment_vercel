---
description: This page contains window functions supported by e6data.
---

# Window Functions

* [Ranking Functions](window-functions.md#ranking-functions)
  * [RANK](window-functions.md#rank)
  * [DENSE\_RANK](window-functions.md#dense\_rank)
  * [ROW\_NUMBER](window-functions.md#row\_number)
  * [NTILE](window-functions.md#ntile-n)
* [Value Functions](window-functions.md#value-functions)
  * [FIRST\_VALUE](window-functions.md#first\_value-less-than-expr-greater-than)
  * [LAST\_VALUE](window-functions.md#last\_value-less-than-expr-greater-than)
  * [LEAD](window-functions.md#lead-less-than-expr-greater-than-offset-default\_value)
  * [LAG](window-functions.md#lag-less-than-expr-greater-than-offset-default\_val)
* [Window Frame Functions](window-functions.md#window-frame-functions)
* [Other Functions](window-functions.md#other-functions)
  * [COLLECT\_LIST](window-functions.md#collect\_list)

### Ranking Functions <a href="#ranking-functions" id="ranking-functions"></a>

#### <mark style="color:purple;">RANK()</mark>

Returns the rank of a value compared to all values in the partition. _**It will produce gaps**_ in the ranking sequence and does not break ties.

```sql
rank() over(order by a)
rank() over(partition by a order by b)
```

#### <mark style="color:purple;">DENSE\_RANK()</mark>

Returns the rank of a value compared to all values in the partition. _**It will **<mark style="color:blue;">**not**</mark>** produce gaps**_ in the ranking sequence and does not break ties.

```sql
dense_rank() over(order by a)
dense_rank() over(partition by a order by b)
```

#### <mark style="color:purple;">ROW\_NUMBER()</mark>

Returns a unique, sequential number for each row, starting with `1`, according to the ordering of rows within the window partition

```sql
row_number() over(order by a)
row_number() over(partition by a order by b)
```

#### <mark style="color:purple;">NTILE(</mark>  n  <mark style="color:purple;">)</mark>

Divides the rows for each window partition into `n` , where n is greater than 1 bucket, ranging from `1` to at most `n`

```sql
ntile(n) over(order by a)
ntile(n) over(partition by a order by b)
```

<pre class="language-sql"><code class="lang-sql"><strong>> select colA, ntile(10) over( order by colA desc) as ntile_val from table;
</strong>
> select colB, ntile(10) over( partition by colB order by colc desc) as ntile_val from table;
</code></pre>

### Value Functions

#### <mark style="color:purple;">FIRST\_VALUE(</mark>  \<expr> <mark style="color:purple;">)</mark>

Returns the first value of `expr` for a group of rows.

```sql
> SELECT first_value(col) over (order by colA desc) FROM table;
```

#### <mark style="color:purple;">LAST\_VALUE(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the last value of `expr` for a group of rows.

```sql
> SELECT last_value(col) over ( partition by colB order by colA desc) FROM table;
```

#### <mark style="color:purple;">LEAD(</mark>  \<expr>,  offset  <mark style="color:purple;">)</mark>

Returns data in a subsequent row in the same result set without joining the table to itself.

If the offset is null or larger than the window NULL is returned.

```sql
> select * from (
    select lead(count(colB),5) over (order by colA)
    from table)
  order by colA DESC;
```

#### <mark style="color:purple;">LAG(</mark>  \<expr>,  offset  <mark style="color:purple;">)</mark>

Returns data in a preceding row in the same result set without joining the table to itself.

If the offset is null or larger than the window NULL is returned.

```sql
> select * from (
    select lag(count(colB),5) over (order by colA)
    from table)
  order by colA DESC;
```

### Window Frame Functions

Window Frame functions provide the ability to specify the upper and lower bounds of a window frame.

_Note: e6data currently supports only SUM & COUNT functions in Window Frame functions_

Syntax:

{% code overflow="wrap" %}
```sql
<function> ( <arguments> ) OVER ( PARTITION BY <expr1> ORDER BY <expr2> 
<frame_type> )
```
{% endcode %}

* Cumulative Frames

```sql
rows between unbounded preceding and current row
rows between current row and unbounded following
```

* Sliding Frames

```sql
rows between <num> { preceding | following } and <num> { preceding | following }
rows between unbounded preceding and <num> { preceding | following }
rows between <num> { preceding | following } and unbounded following
```

```sql
> select
    count(a) over (partition by b order by c rows between unbounded preceding and current row),
    count(a) over (partition by b order by c rows between 2 following and  unbounded following),
    count(a) over (partition by b order by c rows between 2 preceding and  2 following),
    sum(a)   over (partition by b order by c rows between unbounded preceding and current row),
    sum(a) over (partition by b order by c rows between 2 following and  unbounded following),
    sum(a) over (partition by b order by c rows between 2 preceding and  2 following)
  from table
  ;
```

### Other Functions

#### <mark style="color:purple;">COLLECT\_LIST()</mark>

* The collect\_list function enables you to create an ordered list of values from a specified column within a group or window.
* It is particularly useful when you need to gather and process multiple values associated with a particular grouping key or window frame.

#### Usage Examples:

1. **Aggregation Example:** This query below aggregates values from the "value\_column" and creates a list of those values for each unique "key\_column" value.

```sql
SELECT key_column, collect_list(value_column) AS value_list
FROM table_name
GROUP BY key_column;
```

2. **Windowing Example:** This query generates a list of values from the "value\_column" within each window defined by the "partition\_column." The values are ordered by the "order\_column" within each window.

```sql
SELECT key_column, collect_list(value_column) OVER (PARTITION BY partition_column ORDER BY order_column) AS value_list
FROM table_name;
```

#### Limitations:

1. Unsupported Complex Data Types: The collect\_list function currently does not support complex data type columns as input. It is designed to work with simple data types such as integers, strings, or dates.
2. Absence of Sorting Capability: The collect\_list function does not allow for sorting operations, such as using the ORDER BY clause. The resulting list is generated based on the order of appearance of the values in the dataset, without the ability to explicitly control the ordering.
3. Incompatibility with Distinct Operation: The collect\_list function cannot be used in conjunction with the DISTINCT operation. It is intended to gather all values within a specified grouping or windowing context, without eliminating duplicates or applying distinctness.
