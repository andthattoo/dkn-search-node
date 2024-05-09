pub mod scraper;
pub mod finance;
pub mod search_ddg;
pub mod search_google;


pub use self::finance::StockScraper;
pub use self::search_ddg::DDGSearcher;
pub use self::scraper::Scraper;
