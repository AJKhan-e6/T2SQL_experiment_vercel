# Timezone Functions

#### <mark style="color:purple;">CURRENT\_DATE(</mark>  \<timezone>  <mark style="color:purple;">)</mark>

Returns the current date in a given timezone at the start of a query.

```sql
> select current_date('UTC-11');
2023-01-24
```

#### <mark style="color:purple;">CURRENT\_TIMESTAMP(</mark>  \<timezone>  <mark style="color:purple;">)</mark>

Returns the current date & time in a given timezone at the start of a query, in the `yyyy-mm-dd hh:mm:ss` timestamp format.

```sql
> select current_timestamp('America/Anchorage');
2023-01-25 08:08:00
```

#### <mark style="color:purple;">DATETIME(</mark>  \<expr>  \[,  \<timezone>]  <mark style="color:purple;">)</mark>

Returns the timestamp expression formatted to the given `timezone`. The default timezone is UTC.

```sql
> select datetime(datetime(current_timestamp, 'UTC'), 'Pacific/Samoa')
'2023-01-05 22:27:20'
```
