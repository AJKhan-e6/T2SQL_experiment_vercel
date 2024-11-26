# Format Functions

This section contains format functions that uses the pattern. The table below contains the supported patterns for the format functions.

| Letter for Pattern     | Date or Time component          |
| ---------------------- | ------------------------------- |
| G                      | Era designator                  |
| y or yy or yyyy        | Year                            |
| M or MM or MMM or MMMM | Month in year                   |
| w                      | Results in week in year         |
| W                      | Results in week in month        |
| D                      | Gives the day count in the year |
| d                      | Day of the month                |
| F                      | Day of the week in month        |
| E                      | Day name in the week            |
| a                      | AM or PM marker                 |
| H                      | Hour in the day (0-23)          |
| k                      | Hour in the day (1-24)          |
| m                      | Minute in the hour              |
| s                      | Second in the minute            |

#### <mark style="color:purple;">FORMAT\_DATE(</mark>  \<expr>,  format  <mark style="color:purple;">)</mark>

Returns the date expression formatted to the specified format.

```sql
> select format_date(cast('2023-02-15' as date), 'MM'))
02
```

#### <mark style="color:purple;">FORMAT\_TIMESTAMP(</mark>  \<expr>,  format  <mark style="color:purple;">)</mark>

Return the timestamp formatted according to the specified format

```sql
> select format_timestamp(current_timestamp(), 'mm');
2023
```

