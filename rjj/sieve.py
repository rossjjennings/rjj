class Sieve:
    def __init__(self):
        self.known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def primes(self):
        """
        Generates primes forever.
        """
        for p in self.known_primes:
            yield p
        yield from self.new_primes()

    def new_primes(self):
        """
        Generates primes not seen before.
        """
        n = 30*(self.known_primes[-1]//30 + 1)
        while True:
            for k in [1, 7, 11, 13, 17, 19, 23, 29]:
                is_prime = True
                for p in self.known_primes:
                    if p**2 > n + k:
                        break
                    if (n + k) % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    self.known_primes.append(n + k)
                    yield n + k
            n += 30

    def get_prime(self, n):
        """
        Returns the nth prime number.
        """
        generator = self.new_primes()
        while len(self.known_primes) < n:
            p = next(generator)
        return self.known_primes[n-1]

    def next_prime(self, n):
        """
        Returns the smallest prime number greater than n.
        """
        if n < self.known_primes[-1]:
            # do a bisection search through the known primes
            lower = 0
            upper = len(self.known_primes)-1
            mid = (lower + upper)//2
            while True:
                p_lower = self.known_primes[lower]
                p_upper = self.known_primes[upper]
                p_mid = self.known_primes[mid]
                p_next = self.known_primes[mid + 1]
                if n < p_mid:
                    upper = mid
                    mid = (lower + upper)//2
                elif n > p_next:
                    lower = mid + 1
                    mid = (lower + upper)//2
                elif n == p_next:
                    return self.known_primes[mid + 2]
                else:
                    return p_next
        else:
            # generate new primes until we find it
            generator = new_primes()
            while self.known_primes[-1] <= n:
                p = next(generator)
            return self.known_primes[-1]
