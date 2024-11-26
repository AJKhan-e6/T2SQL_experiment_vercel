---
description: This page contains the Comparison operators supported by e6data.
---

# Comparison Operators & Functions

| Operator |      Description      |
| :------: | :-------------------: |
|     <    |       less than       |
|     >    |      greater than     |
|     =    |         equals        |
|    <>    |       not equals      |
|    <=    |   less than equal to  |
|    >=    | greater than equal to |

#### <mark style="color:purple;">IS NULL</mark>

Returns boolean when the value equates to null.

```sql
> select * from table where col1 IS NULL; -- true/false
```

#### <mark style="color:purple;">IS NOT NULL</mark>

Returns boolean when the value does not equate to null.

```sql
> select * from table where col1 IS NOT NULL; -- true/false
```

#### <mark style="color:purple;">IN(</mark>  value1,  value2,  value3...<mark style="color:purple;">)</mark>

```sql
WHERE element IN (expr1, expr2, expr3...)
```

Returns a boolean value **true** if the `element` value is present in the given set of expressions. Else false is returned.

```sql
> select colA from Table1 where colB in (val1, val2, val3);
```
