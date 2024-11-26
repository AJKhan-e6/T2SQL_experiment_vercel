---
description: >-
  This page contains the Mathematical functions and operators supported by
  e6data.
---

# Mathematical Functions & Operators

### Mathematical Operators

| Operator |   Description  |
| :------: | :------------: |
|     +    |    Addition    |
|     -    |   Subtraction  |
|    \*    | Multiplication |
|     /    |    Division    |

### Mathematical Functions

* [CEIL](mathematical-functions-and-operators.md#ceil-less-than-expr-greater-than)
* [FLOOR](mathematical-functions-and-operators.md#floor-less-than-expr-greater-than)
* [ROUND](mathematical-functions-and-operators.md#round-less-than-expr-greater-than-less-than-scale-greater-than)
* [ABS](mathematical-functions-and-operators.md#abs-less-than-expr-greater-than)
* [POWER](mathematical-functions-and-operators.md#power-less-than-expr1-greater-than-less-than-expr2-greater-than)

#### <mark style="color:purple;">CEIL(</mark>  \<expr>  <mark style="color:purple;">)</mark>&#x20;

Returns the smallest integer which is greater than or equal to the specified numerical expression.

```sql
> select ceil(2.34);
3
```

#### <mark style="color:purple;">FLOOR(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the largest integer which is less than or equal to the specified numerical expression.

```sql
> select floor(6.34);
6
```

#### <mark style="color:purple;">ROUND(</mark>  \<expr>  \[,  \<scale>  ]  <mark style="color:purple;">)</mark> &#x20;

Returns the numeric value, rounded off to a given number of decimal places, of the input numeric expression. If the scale factor is not specified, it will round off to the nearest integer.

`scale` - optional numeric expression, representing the number of decimal places to be rounded off to in a given numeric expression. Default is 0.

```sql
> select round(5.678);
6
> select round(5.678, 2);
5.68
```

#### <mark style="color:purple;">ABS(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the absolute value of the specified numeric expression.

```sql
select abs(colB) from table;

> select abs(-100);
100
```

#### <mark style="color:purple;">POWER(</mark>  \<expr1>,   \<expr2>  <mark style="color:purple;">)</mark> &#x20;

Returns the value of  `expr1` raised to the power of `expr2.`

```sql
> select power(3.15, 0.5);
1.775
```
