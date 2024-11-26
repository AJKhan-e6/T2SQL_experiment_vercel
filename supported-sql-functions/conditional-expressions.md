# Conditional Expressions

* [CASE](conditional-expressions.md#case)
* [COALESCE](conditional-expressions.md#coalesce)
* [NULLIF](conditional-expressions.md#nullif-value1-value2)

#### <mark style="color:purple;">CASE</mark>

Returns the result of the first condition evaluating to true or returns the default value if all conditions evaluate to false.&#x20;

```sql
SYNTAX - CASE
             WHEN condition1 THEN result1
             [WHEN condition2 THEN result2 ]
             [WHEN ......]
             [ELSE default ]
         END
```

```sql
> select col1,
               case
                    when col1 < 0 then 'res1'
                    when col1 > 0 then 'res2'
                    else 'res3'
               end as result
           from table
```

#### <mark style="color:purple;">COALESCE</mark>

Returns the first non-null value from a list of expressions.

```sql
coalesce(val1, val2, val3 ......)
```

```sql
> select coalesce(NULL,20,NULL,NULL,'String1');
20
```

#### <mark style="color:purple;">NULLIF(</mark>  value1,  value2  <mark style="color:purple;">)</mark>

Returns null if `value1` equals `value2`, otherwise returns `value1`

```sql
> select nullif(24, 24);
null

> select nullif(12, null);
12
```
