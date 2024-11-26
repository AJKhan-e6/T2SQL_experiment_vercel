---
description: Bitwise functions supported by e6data
---

# Bitwise Functions

#### <mark style="color:purple;">SHIFTRIGHT (</mark> \<expr>, \<n> <mark style="color:purple;">)</mark>

Alternate syntax: `<expr>`` `<mark style="color:purple;">`>>`</mark>` ``n;`

Returns a bitwise signed right shifted by `n` bits.

```sql
> SELECT shiftright(2, 1);
1
```

<pre class="language-sql"><code class="lang-sql"><strong>> SELECT 2 >> 1;
</strong>1
</code></pre>

#### <mark style="color:purple;">SHIFTLEFT (</mark> \<expr>, \<n> <mark style="color:purple;">)</mark>

Alternate syntax: `<expr>`` `<mark style="color:purple;">`<<`</mark>` ``n;`

Returns a bitwise signed left shifted by `n` bits.

```sql
> SELECT shiftleft(2, 1);
4
```

```sql
> SELECT 2 << 1;
4
```
