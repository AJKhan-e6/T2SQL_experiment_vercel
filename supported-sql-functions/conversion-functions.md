---
description: This page contains the explicit conversion functions supported by e6data.
---

# Conversion Functions

* [CAST](conversion-functions.md#cast-less-than-expr-greater-than-as-less-than-target-datatype-greater-than)

## CAST FUNCTION

#### <mark style="color:purple;">CAST(</mark> \<expr>  as  \<target datatype>  <mark style="color:purple;">)</mark>

Converts the input expression to the specified target datatype.

```sql
cast('2022-01-11' as date)
```

The following table contains a matrix of all supported conversions:

<table><thead><tr><th width="175">Source(Row) Target(Column)</th><th align="center">varchar</th><th width="115" align="center">integer/int</th><th align="center">bigint</th><th align="center">float</th><th width="100" align="center">double</th><th width="104" align="center">boolean</th><th width="82" align="center">date</th><th width="118" align="center">timestamp</th></tr></thead><tbody><tr><td>varchar</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td></tr><tr><td>integer/int</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td></tr><tr><td>bigint</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr><tr><td>float</td><td align="center">Y</td><td align="center">Y</td><td align="center">-</td><td align="center">Y</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr><tr><td>double</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr><tr><td>boolean</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td></tr><tr><td>date</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">Y</td><td align="center">Y</td></tr><tr><td>timestamp</td><td align="center">Y</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">-</td><td align="center">Y</td><td align="center">Y</td></tr></tbody></table>
