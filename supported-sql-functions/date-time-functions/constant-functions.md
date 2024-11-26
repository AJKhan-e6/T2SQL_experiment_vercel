# Constant Functions

#### <mark style="color:purple;">**CURRENT\_DATE()**</mark>

Returns the current date at the start of a query.

```sql
> select current_date();
2023-01-25
```

#### <mark style="color:purple;">CURRENT\_TIMESTAMP()</mark>

Returns the current date & time at the start of a query, in the `yyyy-mm-dd hh:mm:ss` timestamp format.

```sql
> select current_timestamp;
2023-01-25 08:08:00
```

#### <mark style="color:purple;">NOW()</mark>

Alias of [current\_timestamp()](constant-functions.md#current\_timestamp)
