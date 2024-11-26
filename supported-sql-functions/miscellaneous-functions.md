# Miscellaneous Functions

#### <mark style="color:purple;">EXCEPT (column\_name)</mark>&#x20;

`SELECT * EXCEPT` _columns that should be excluded from the results_

```sql
> WITH cte AS (SELECT 5 as c1,'e6data' as c2, 10 as c3)
  SELECT * EXCEPT (c3, c2) FROM cte

c2
5
```
