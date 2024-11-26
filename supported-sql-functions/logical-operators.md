---
description: This page contains logical operators supported by e6data.
---

# Logical Operators

* [AND](logical-operators.md#less-than-expr1-greater-than-and-less-than-expr2-greater-than)
* [OR](logical-operators.md#less-than-expr1-greater-than-or-less-than-expr2-greater-than)
* [NOT](logical-operators.md#not-less-than-expr-greater-than)

#### \<expr1>  <mark style="color:purple;">AND</mark>  \<expr2>

Returns true if both the values evaluate to true.

```sql
> select colA from table1 where colB > x AND colB <> y;
```

#### \<expr1>  <mark style="color:purple;">OR</mark>  \<expr2>

Returns true if any of the values evaluate to true.

```sql
> select colA from table1 where colB > x OR colB = null;
```

#### <mark style="color:purple;">NOT</mark>  \<expr>

Returns true if the value is false.

```sql
> select colA from table1 where colB NOT y;
```

The following truth table demonstrates the boolean output of `AND` and `OR:`

| a     | b     | a AND b | a OR b |
| ----- | ----- | ------- | ------ |
| true  | true  | true    | true   |
| true  | false | false   | true   |
| false | true  | false   | true   |
| false | false | false   | false  |
