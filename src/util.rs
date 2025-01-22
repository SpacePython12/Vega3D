use serde::Deserialize;


pub enum MaybeBorrowed<'b, B: Sized + 'b> {
    Borrowed(&'b B),
    Owned(B)
}

impl<'b, B> std::ops::Deref for MaybeBorrowed<'b, B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        match self {
            MaybeBorrowed::Borrowed(val) => val,
            MaybeBorrowed::Owned(val) => val,
        }
    }
}

impl<'b, B> MaybeBorrowed<'b, B> {
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        match self {
            MaybeBorrowed::Borrowed(_) => true,
            MaybeBorrowed::Owned(_) => false,
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        !self.is_borrowed()
    }
}

impl<'b, B> From<&'b B> for MaybeBorrowed<'b, B> {
    fn from(value: &'b B) -> Self {
        MaybeBorrowed::Borrowed(value)
    }
}

impl<'b, B> From<B> for MaybeBorrowed<'b, B> 
where B: 'b {
    fn from(value: B) -> Self {
        MaybeBorrowed::Owned(value)
    }
}

impl<'b, B> std::fmt::Debug for MaybeBorrowed<'b, B>
where B: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaybeBorrowed::Borrowed(val) => val.fmt(f),
            MaybeBorrowed::Owned(val) => val.fmt(f),
        }
    }
} 