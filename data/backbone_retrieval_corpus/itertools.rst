:mod:`!itertools` --- Functions creating iterators for efficient looping
========================================================================

.. module:: itertools
   :synopsis: Functions creating iterators for efficient looping.

.. testsetup::

   from itertools import *
   import collections
   import math
   import operator
   import random

--------------

This module implements a number of :term:`iterator` building blocks inspired
by constructs from APL, Haskell, and SML.  Each has been recast in a form
suitable for Python.

The module standardizes a core set of fast, memory efficient tools that are
useful by themselves or in combination.  Together, they form an "iterator
algebra" making it possible to construct specialized tools succinctly and
efficiently in pure Python.

For instance, SML provides a tabulation tool: ``tabulate(f)`` which produces a
sequence ``f(0), f(1), ...``.  The same effect can be achieved in Python
by combining :func:`map` and :func:`count` to form ``map(f, count())``.

**General iterators:**

============================    ============================    =================================================   =============================================================
Iterator                        Arguments                       Results                                             Example
============================    ============================    =================================================   =============================================================
:func:`accumulate`              p [,func]                       p0, p0+p1, p0+p1+p2, ...                            ``accumulate([1,2,3,4,5]) → 1 3 6 10 15``
:func:`batched`                 p, n                            (p0, p1, ..., p_n-1), ...                           ``batched('ABCDEFG', n=3) → ABC DEF G``
:func:`chain`                   p, q, ...                       p0, p1, ... plast, q0, q1, ...                      ``chain('ABC', 'DEF') → A B C D E F``
:func:`chain.from_iterable`     iterable                        p0, p1, ... plast, q0, q1, ...                      ``chain.from_iterable(['ABC', 'DEF']) → A B C D E F``
:func:`compress`                data, selectors                 (d[0] if s[0]), (d[1] if s[1]), ...                 ``compress('ABCDEF', [1,0,1,0,1,1]) → A C E F``
:func:`count`                   [start[, step]]                 start, start+step, start+2*step, ...                ``count(10) → 10 11 12 13 14 ...``
:func:`cycle`                   p                               p0, p1, ... plast, p0, p1, ...                      ``cycle('ABCD') → A B C D A B C D ...``
:func:`dropwhile`               predicate, seq                  seq[n], seq[n+1], starting when predicate fails     ``dropwhile(lambda x: x<5, [1,4,6,3,8]) → 6 3 8``
:func:`filterfalse`             predicate, seq                  elements of seq where predicate(elem) fails         ``filterfalse(lambda x: x<5, [1,4,6,3,8]) → 6 8``
:func:`groupby`                 iterable[, key]                 sub-iterators grouped by value of key(v)            ``groupby(['A','B','DEF'], len) → (1, A B) (3, DEF)``
:func:`islice`                  seq, [start,] stop [, step]     elements from seq[start:stop:step]                  ``islice('ABCDEFG', 2, None) → C D E F G``
:func:`pairwise`                iterable                        (p[0], p[1]), (p[1], p[2])                          ``pairwise('ABCDEFG') → AB BC CD DE EF FG``
:func:`repeat`                  elem [,n]                       elem, elem, elem, ... endlessly or up to n times    ``repeat(10, 3) → 10 10 10``
:func:`starmap`                 func, seq                       func(\*seq[0]), func(\*seq[1]), ...                 ``starmap(pow, [(2,5), (3,2), (10,3)]) → 32 9 1000``
:func:`takewhile`               predicate, seq                  seq[0], seq[1], until predicate fails               ``takewhile(lambda x: x<5, [1,4,6,3,8]) → 1 4``
:func:`tee`                     it, n                           it1, it2, ... itn  splits one iterator into n       ``tee('ABC', 2) → A B C, A B C``
:func:`zip_longest`             p, q, ...                       (p[0], q[0]), (p[1], q[1]), ...                     ``zip_longest('ABCD', 'xy', fillvalue='-') → Ax By C- D-``
============================    ============================    =================================================   =============================================================

**Combinatoric iterators:**

==============================================   ====================       =============================================================
Iterator                                         Arguments                  Results
==============================================   ====================       =============================================================
:func:`product`                                  p, q, ... [repeat=1]       cartesian product, equivalent to a nested for-loop
:func:`permutations`                             p[, r]                     r-length tuples, all possible orderings, no repeated elements
:func:`combinations`                             p, r                       r-length tuples, in sorted order, no repeated elements
:func:`combinations_with_replacement`            p, r                       r-length tuples, in sorted order, with repeated elements
==============================================   ====================       =============================================================

==============================================   =============================================================
Examples                                         Results
==============================================   =============================================================
``product('ABCD', repeat=2)``                    ``AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD``
``permutations('ABCD', 2)``                      ``AB AC AD BA BC BD CA CB CD DA DB DC``
``combinations('ABCD', 2)``                      ``AB AC AD BC BD CD``
``combinations_with_replacement('ABCD', 2)``     ``AA AB AC AD BB BC BD CC CD DD``
==============================================   =============================================================


.. _itertools-functions:

Itertool Functions
------------------

The following functions all construct and return iterators. Some provide
streams of infinite length, so they should only be accessed by functions or
loops that truncate the stream.


.. function:: accumulate(iterable[, function, *, initial=None])

    Make an iterator that returns accumulated sums or accumulated
    results from other binary functions.

    The *function* defaults to addition.  The *function* should accept
    two arguments, an accumulated total and a value from the *iterable*.

    If an *initial* value is provided, the accumulation will start with
    that value and the output will have one more element than the input
    iterable.

    Roughly equivalent to::

        def accumulate(iterable, function=operator.add, *, initial=None):
            'Return running totals'
            # accumulate([1,2,3,4,5]) → 1 3 6 10 15
            # accumulate([1,2,3,4,5], initial=100) → 100 101 103 106 110 115
            # accumulate([1,2,3,4,5], operator.mul) → 1 2 6 24 120

            iterator = iter(iterable)
            total = initial
            if initial is None:
                try:
                    total = next(iterator)
                except StopIteration:
                    return

            yield total
            for element in iterator:
                total = function(total, element)
                yield total

    To compute a running minimum, set *function* to :func:`min`.
    For a running maximum, set *function* to :func:`max`.
    Or for a running product, set *function* to :func:`operator.mul`.
    To build an `amortization table
    <https://www.ramseysolutions.com/real-estate/amortization-schedule>`_,
    accumulate the interest and apply payments:

    .. doctest::

      >>> data = [3, 4, 6, 2, 1, 9, 0, 7, 5, 8]
      >>> list(accumulate(data, max))              # running maximum
      [3, 4, 6, 6, 6, 9, 9, 9, 9, 9]
      >>> list(accumulate(data, operator.mul))     # running product
      [3, 12, 72, 144, 144, 1296, 0, 0, 0, 0]

      # Amortize a 5% loan of 1000 with 10 annual payments of 90
      >>> update = lambda balance, payment: round(balance * 1.05) - payment
      >>> list(accumulate(repeat(90, 10), update, initial=1_000))
      [1000, 960, 918, 874, 828, 779, 728, 674, 618, 559, 497]

    See :func:`functools.reduce` for a similar function that returns only the
    final accumulated value.

    .. versionadded:: 3.2

    .. versionchanged:: 3.3
       Added the optional *function* parameter.

    .. versionchanged:: 3.8
       Added the optional *initial* parameter.


.. function:: batched(iterable, n, *, strict=False)

   Batch data from the *iterable* into tuples of length *n*. The last
   batch may be shorter than *n*.

   If *strict* is true, will raise a :exc:`ValueError` if the final
   batch is shorter than *n*.

   Loops over the input iterable and accumulates data into tuples up to
   size *n*.  The input is consumed lazily, just enough to fill a batch.
   The result is yielded as soon as the batch is full or when the input
   iterable is exhausted:

   .. doctest::

      >>> flattened_data = ['roses', 'red', 'violets', 'blue', 'sugar', 'sweet']
      >>> unflattened = list(batched(flattened_data, 2))
      >>> unflattened
      [('roses', 'red'), ('violets', 'blue'), ('sugar', 'sweet')]

   Roughly equivalent to::

      def batched(iterable, n, *, strict=False):
          # batched('ABCDEFG', 3) → ABC DEF G
          if n < 1:
              raise ValueError('n must be at least one')
          iterator = iter(iterable)
          while batch := tuple(islice(iterator, n)):
              if strict and len(batch) != n:
                  raise ValueError('batched(): incomplete batch')
              yield batch

   .. versionadded:: 3.12

   .. versionchanged:: 3.13
      Added the *strict* option.


.. function:: chain(*iterables)

   Make an iterator that returns elements from the first iterable until
   it is exhausted, then proceeds to the next iterable, until all of the
   iterables are exhausted.  This combines multiple data sources into a
   single iterator.  Roughly equivalent to::

      def chain(*iterables):
          # chain('ABC', 'DEF') → A B C D E F
          for iterable in iterables:
              yield from iterable

   Note that :ref:`unpacking in comprehensions <unpacking-comprehensions>`
   provides similar functionality so that ``list(chain(p, q))`` could be
   written as ``[*s for s in (p, q)]``.


.. classmethod:: chain.from_iterable(iterable)

   Alternate constructor for :func:`chain`.  Gets chained inputs from a
   single iterable argument that is evaluated lazily.  Roughly equivalent to::

      def from_iterable(iterables):
          # chain.from_iterable(['ABC', 'DEF']) → A B C D E F
          for iterable in iterables:
              yield from iterable

   Note that :ref:`unpacking in comprehensions <unpacking-comprehensions>`
   provides similar functionality so that ``list(chain.from_iterable(iterables))``
   could be written as ``[*s for s in iterables]``.


.. function:: combinations(iterable, r)

   Return *r* length subsequences of elements from the input *iterable*.

   The output is a subsequence of :func:`product` keeping only entries that
   are subsequences of the *iterable*.  The length of the output is given
   by :func:`math.comb` which computes ``n! / r! / (n - r)!`` when ``0 ≤ r
   ≤ n`` or zero when ``r > n``.

   The combination tuples are emitted in lexicographic order according to
   the order of the input *iterable*. If the input *iterable* is sorted,
   the output tuples will be produced in sorted order.

   Elements are treated as unique based on their position, not on their
   value.  If the input elements are unique, there will be no repeated
   values within each combination.

   Roughly equivalent to::

        def combinations(iterable, r):
            # combinations('ABCD', 2) → AB AC AD BC BD CD
            # combinations(range(4), 3) → 012 013 023 123

            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))

            yield tuple(pool[i] for i in indices)
            while True:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)


.. function:: combinations_with_replacement(iterable, r)

   Return *r* length subsequences of elements from the input *iterable*
   allowing individual elements to be repeated more than once.

   The output is a subsequence of :func:`product` that keeps only entries
   that are subsequences (with possible repeated elements) of the
   *iterable*.  The number of subsequence returned is ``(n + r - 1)! / r! /
   (n - 1)!`` when ``n > 0``.

   The combination tuples are emitted in lexicographic order according to
   the order of the input *iterable*. if the input *iterable* is sorted,
   the output tuples will be produced in sorted order.

   Elements are treated as unique based on their position, not on their
   value.  If the input elements are unique, the generated combinations
   will also be unique.

   Roughly equivalent to::

        def combinations_with_replacement(iterable, r):
            # combinations_with_replacement('ABC', 2) → AA AB AC BB BC CC

            pool = tuple(iterable)
            n = len(pool)
            if not n and r:
                return
            indices = [0] * r

            yield tuple(pool[i] for i in indices)
            while True:
                for i in reversed(range(r)):
                    if indices[i] != n - 1:
                        break
                else:
                    return
                indices[i:] = [indices[i] + 1] * (r - i)
                yield tuple(pool[i] for i in indices)

   .. versionadded:: 3.1


.. function:: compress(data, selectors)

   Make an iterator that returns elements from *data* where the
   corresponding element in *selectors* is true.  Stops when either the
   *data* or *selectors* iterables have been exhausted.  Roughly
   equivalent to::

       def compress(data, selectors):
           # compress('ABCDEF', [1,0,1,0,1,1]) → A C E F
           return (datum for datum, selector in zip(data, selectors) if selector)

   .. versionadded:: 3.1


.. function:: count(start=0, step=1)

   Make an iterator that returns evenly spaced values beginning with
   *start*. Can be used with :func:`map` to generate consecutive data
   points or with :func:`zip` to add sequence numbers.  Roughly
   equivalent to::

      def count(start=0, step=1):
          # count(10) → 10 11 12 13 14 ...
          # count(2.5, 0.5) → 2.5 3.0 3.5 ...
          n = start
          while True:
              yield n
              n += step

   When counting with floating-point numbers, better accuracy can sometimes be
   achieved by substituting multiplicative code such as: ``(start + step * i
   for i in count())``.

   .. versionchanged:: 3.1
      Added *step* argument and allowed non-integer arguments.


.. function:: cycle(iterable)

   Make an iterator returning elements from the *iterable* and saving a
   copy of each.  When the iterable is exhausted, return elements from
   the saved copy.  Repeats indefinitely.  Roughly equivalent to::

      def cycle(iterable):
          # cycle('ABCD') → A B C D A B C D A B C D ...

          saved = []
          for element in iterable:
              yield element
              saved.append(element)

          while saved:
              for element in saved:
                  yield element

   This itertool may require significant auxiliary storage (depending on
   the length of the iterable).


.. function:: dropwhile(predicate, iterable)

   Make an iterator that drops elements from the *iterable* while the
   *predicate* is true and afterwards returns every element.  Roughly
   equivalent to::

      def dropwhile(predicate, iterable):
          # dropwhile(lambda x: x<5, [1,4,6,3,8]) → 6 3 8

          iterator = iter(iterable)
          for x in iterator:
              if not predicate(x):
                  yield x
                  break

          for x in iterator:
              yield x

   Note this does not produce *any* output until the predicate first
   becomes false, so this itertool may have a lengthy start-up time.


.. function:: filterfalse(predicate, iterable)

   Make an iterator that filters elements from the *iterable* returning
   only those for which the *predicate* returns a false value.  If
   *predicate* is ``None``, returns the items that are false.  Roughly
   equivalent to::

      def filterfalse(predicate, iterable):
          # filterfalse(lambda x: x<5, [1,4,6,3,8]) → 6 8

          if predicate is None:
              predicate = bool

          for x in iterable:
              if not predicate(x):
                  yield x


.. function:: groupby(iterable, key=None)

   Make an iterator that returns consecutive keys and groups from the *iterable*.
   The *key* is a function computing a key value for each element.  If not
   specified or is ``None``, *key* defaults to an identity function and returns
   the element unchanged.  Generally, the iterable needs to already be sorted on
   the same key function.

   The operation of :func:`groupby` is similar to the ``uniq`` filter in Unix.  It
   generates a break or new group every time the value of the key function changes
   (which is why it is usually necessary to have sorted the data using the same key
   function).  That behavior differs from SQL's GROUP BY which aggregates common
   elements regardless of their input order.

   The returned group is itself an iterator that shares the underlying iterable
   with :func:`groupby`.  Because the source is shared, when the :func:`groupby`
   object is advanced, the previous group is no longer visible.  So, if that data
   is needed later, it should be stored as a list::

      groups = []
      uniquekeys = []
      data = sorted(data, key=keyfunc)
      for k, g in groupby(data, keyfunc):
          groups.append(list(g))      # Store group iterator as a list
          uniquekeys.append(k)

   :func:`groupby` is roughly equivalent to::

      def groupby(iterable, key=None):
          # [k for k, g in groupby('AAAABBBCCDAABBB')] → A B C D A B
          # [list(g) for k, g in groupby('AAAABBBCCD')] → AAAA BBB CC D

          keyfunc = (lambda x: x) if key is None else key
          iterator = iter(iterable)
          exhausted = False

          def _grouper(target_key):
              nonlocal curr_value, curr_key, exhausted
              yield curr_value
              for curr_value in iterator:
                  curr_key = keyfunc(curr_value)
                  if curr_key != target_key:
                      return
                  yield curr_value
              exhausted = True

          try:
              curr_value = next(iterator)
          except StopIteration:
              return
          curr_key = keyfunc(curr_value)

          while not exhausted:
              target_key = curr_key
              curr_group = _grouper(target_key)
              yield curr_key, curr_group
              if curr_key == target_key:
                  for _ in curr_group:
                      pass


.. function:: islice(iterable, stop)
              islice(iterable, start, stop[, step])

   Make an iterator that returns selected elements from the iterable.
   Works like sequence slicing but does not support negative values for
   *start*, *stop*, or *step*.

   If *start* is zero or ``None``, iteration starts at zero.  Otherwise,
   elements from the iterable are skipped until *start* is reached.

   If *stop* is ``None``, iteration continues until the input is
   exhausted, if at all.  Otherwise, it stops at the specified position.

   If *step* is ``None``, the step defaults to one.  Elements are returned
   consecutively unless *step* is set higher than one which results in
   items being skipped.

   Roughly equivalent to::

      def islice(iterable, *args):
          # islice('ABCDEFG', 2) → A B
          # islice('ABCDEFG', 2, 4) → C D
          # islice('ABCDEFG', 2, None) → C D E F G
          # islice('ABCDEFG', 0, None, 2) → A C E G

          s = slice(*args)
          start = 0 if s.start is None else s.start
          stop = s.stop
          step = 1 if s.step is None else s.step
          if start < 0 or (stop is not None and stop < 0) or step <= 0:
              raise ValueError

          indices = count() if stop is None else range(max(start, stop))
          next_i = start
          for i, element in zip(indices, iterable):
              if i == next_i:
                  yield element
                  next_i += step

   If the input is an iterator, then fully consuming the *islice*
   advances the input iterator by ``max(start, stop)`` steps regardless
   of the *step* value.


.. function:: pairwise(iterable)

   Return successive overlapping pairs taken from the input *iterable*.

   The number of 2-tuples in the output iterator will be one fewer than the
   number of inputs.  It will be empty if the input iterable has fewer than
   two values.

   Roughly equivalent to::

        def pairwise(iterable):
            # pairwise('ABCDEFG') → AB BC CD DE EF FG

            iterator = iter(iterable)
            a = next(iterator, None)

            for b in iterator:
                yield a, b
                a = b

   .. versionadded:: 3.10


.. function:: permutations(iterable, r=None)

   Return successive *r* length `permutations of elements
   <https://www.britannica.com/science/permutation>`_ from the *iterable*.

   If *r* is not specified or is ``None``, then *r* defaults to the length
   of the *iterable* and all possible full-length permutations
   are generated.

   The output is a subsequence of :func:`product` where entries with
   repeated elements have been filtered out.  The length of the output is
   given by :func:`math.perm` which computes ``n! / (n - r)!`` when
   ``0 ≤ r ≤ n`` or zero when ``r > n``.

   The permutation tuples are emitted in lexicographic order according to
   the order of the input *iterable*.  If the input *iterable* is sorted,
   the output tuples will be produced in sorted order.

   Elements are treated as unique based on their position, not on their
   value.  If the input elements are unique, there will be no repeated
   values within a permutation.

   Roughly equivalent to::

        def permutations(iterable, r=None):
            # permutations('ABCD', 2) → AB AC AD BA BC BD CA CB CD DA DB DC
            # permutations(range(3)) → 012 021 102 120 201 210

            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            if r > n:
                return

            indices = list(range(n))
            cycles = list(range(n, n-r, -1))
            yield tuple(pool[i] for i in indices[:r])

            while n:
                for i in reversed(range(r)):
                    cycles[i] -= 1
                    if cycles[i] == 0:
                        indices[i:] = indices[i+1:] + indices[i:i+1]
                        cycles[i] = n - i
                    else:
                        j = cycles[i]
                        indices[i], indices[-j] = indices[-j], indices[i]
                        yield tuple(pool[i] for i in indices[:r])
                        break
                else:
                    return


.. function:: product(*iterables, repeat=1)

   `Cartesian product <https://en.wikipedia.org/wiki/Cartesian_product>`_
   of the input iterables.

   Roughly equivalent to nested for-loops in a generator expression. For example,
   ``product(A, B)`` returns the same as ``((x,y) for x in A for y in B)``.

   The nested loops cycle like an odometer with the rightmost element advancing
   on every iteration.  This pattern creates a lexicographic ordering so that if
   the input's iterables are sorted, the product tuples are emitted in sorted
   order.

   To compute the product of an iterable with itself, specify the number of
   repetitions with the optional *repeat* keyword argument.  For example,
   ``product(A, repeat=4)`` means the same as ``product(A, A, A, A)``.

   This function is roughly equivalent to the following code, except that the
   actual implementation does not build up intermediate results in memory::

       def product(*iterables, repeat=1):
           # product('ABCD', 'xy') → Ax Ay Bx By Cx Cy Dx Dy
           # product(range(2), repeat=3) → 000 001 010 011 100 101 110 111

           if repeat < 0:
               raise ValueError('repeat argument cannot be negative')
           pools = [tuple(pool) for pool in iterables] * repeat

           result = [[]]
           for pool in pools:
               result = [x+[y] for x in result for y in pool]

           for prod in result:
               yield tuple(prod)

   Before :func:`product` runs, it completely consumes the input iterables,
   keeping pools of values in memory to generate the products.  Accordingly,
   it is only useful with finite inputs.


.. function:: repeat(object[, times])

   Make an iterator that returns *object* over and over again. Runs indefinitely
   unless the *times* argument is specified.

   Roughly equivalent to::

      def repeat(object, times=None):
          # repeat(10, 3) → 10 10 10
          if times is None:
              while True:
                  yield object
          else:
              for i in range(times):
                  yield object

   A common use for *repeat* is to supply a stream of constant values to *map*
   or *zip*:

   .. doctest::

      >>> list(map(pow, range(10), repeat(2)))
      [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


.. function:: starmap(function, iterable)

   Make an iterator that computes the *function* using arguments obtained
   from the *iterable*.  Used instead of :func:`map` when argument
   parameters have already been "pre-zipped" into tuples.

   The difference between :func:`map` and :func:`starmap` parallels the
   distinction between ``function(a,b)`` and ``function(*c)``. Roughly
   equivalent to::

      def starmap(function, iterable):
          # starmap(pow, [(2,5), (3,2), (10,3)]) → 32 9 1000
          for args in iterable:
              yield function(*args)


.. function:: takewhile(predicate, iterable)

   Make an iterator that returns elements from the *iterable* as long as
   the *predicate* is true.  Roughly equivalent to::

      def takewhile(predicate, iterable):
          # takewhile(lambda x: x<5, [1,4,6,3,8]) → 1 4
          for x in iterable:
              if not predicate(x):
                  break
              yield x

   Note, the element that first fails the predicate condition is
   consumed from the input iterator and there is no way to access it.
   This could be an issue if an application wants to further consume the
   input iterator after *takewhile* has been run to exhaustion.  To work
   around this problem, consider using `more-itertools before_and_after()
   <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.before_and_after>`_
   instead.


.. function:: tee(iterable, n=2)

   Return *n* independent iterators from a single iterable.

   Roughly equivalent to::

        def tee(iterable, n=2):
            if n < 0:
                raise ValueError
            if n == 0:
                return ()
            iterator = _tee(iterable)
            result = [iterator]
            for _ in range(n - 1):
                result.append(_tee(iterator))
            return tuple(result)

        class _tee:

            def __init__(self, iterable):
                it = iter(iterable)
                if isinstance(it, _tee):
                    self.iterator = it.iterator
                    self.link = it.link
                else:
                    self.iterator = it
                    self.link = [None, None]

            def __iter__(self):
                return self

            def __next__(self):
                link = self.link
                if link[1] is None:
                    link[0] = next(self.iterator)
                    link[1] = [None, None]
                value, self.link = link
                return value

   When the input *iterable* is already a tee iterator object, all
   members of the return tuple are constructed as if they had been
   produced by the upstream :func:`tee` call.  This "flattening step"
   allows nested :func:`tee` calls to share the same underlying data
   chain and to have a single update step rather than a chain of calls.

   The flattening property makes tee iterators efficiently peekable:

   .. testcode::

      def lookahead(tee_iterator):
           "Return the next value without moving the input forward"
           [forked_iterator] = tee(tee_iterator, 1)
           return next(forked_iterator)

   .. doctest::

      >>> iterator = iter('abcdef')
      >>> [iterator] = tee(iterator, 1)   # Make the input peekable
      >>> next(iterator)                  # Move the iterator forward
      'a'
      >>> lookahead(iterator)             # Check next value
      'b'
      >>> next(iterator)                  # Continue moving forward
      'b'

   ``tee`` iterators are not threadsafe. A :exc:`RuntimeError` may be
   raised when simultaneously using iterators returned by the same :func:`tee`
   call, even if the original *iterable* is threadsafe.

   This itertool may require significant auxiliary storage (depending on how
   much temporary data needs to be stored). In general, if one iterator uses
   most or all of the data before another iterator starts, it is faster to use
   :func:`list` instead of :func:`tee`.


.. function:: zip_longest(*iterables, fillvalue=None)

   Make an iterator that aggregates elements from each of the
   *iterables*.

   If the iterables are of uneven length, missing values are filled-in
   with *fillvalue*.  If not specified, *fillvalue* defaults to ``None``.

   Iteration continues until the longest iterable is exhausted.

   Roughly equivalent to::

      def zip_longest(*iterables, fillvalue=None):
          # zip_longest('ABCD', 'xy', fillvalue='-') → Ax By C- D-

          iterators = list(map(iter, iterables))
          num_active = len(iterators)
          if not num_active:
              return

          while True:
              values = []
              for i, iterator in enumerate(iterators):
                  try:
                      value = next(iterator)
                  except StopIteration:
                      num_active -= 1
                      if not num_active:
                          return
                      iterators[i] = repeat(fillvalue)
                      value = fillvalue
                  values.append(value)
              yield tuple(values)

   If one of the iterables is potentially infinite, then the :func:`zip_longest`
   function should be wrapped with something that limits the number of calls
   (for example :func:`islice` or :func:`takewhile`).


