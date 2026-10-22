use std::collections::HashMap;

use crate::client::{RequestInfo, Response};

pub enum InterceptAction {
    Continue,
    Block,
    Fulfill(Response),
    ModifyHeaders(HashMap<String, String>),
}

#[async_trait::async_trait]
pub trait RequestInterceptor {
    async fn intercept(&self, request: &RequestInfo) -> InterceptAction;

    /// Whether this interceptor only ever acts on document requests. The
    /// shared subresource cache, off while any other interceptor is
    /// installed, then stays on: documents are never cached anyway.
    fn documents_only(&self) -> bool {
        false
    }
}
