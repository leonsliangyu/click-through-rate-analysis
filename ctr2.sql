
use ctr;

CREATE TABLE IF NOT EXISTS ctr.transactions_filtered (
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,
	payment_time			DATETIME,
    money					int,
    kind_pay            	varchar(5),
    kind_card				varchar(7),
    store_id				varchar(10),
    network					varchar(8),
    industry				int,
    gender					varchar(7),
    address					nvarchar(100),
    PRIMARY KEY (user_id, payment_time)
);

INSERT INTO transactions_filtered (user_id, payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, address)
select * from transactions where 
(year(payment_time)=2017 and month(payment_time)=7  and day(payment_time) = 22) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 1) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 20) or
(year(payment_time)=2017 and month(payment_time)=9  and day(payment_time) = 1);



CREATE TABLE IF NOT EXISTS ctr.views_filtered (
	view_time				DATETIME,
    payment_time			DATETIME,
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,		
    store_id				varchar(10),
	ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (user_id, payment_time)
);

INSERT INTO views_filtered(view_time, payment_time, user_id, store_id, ad_id) 
select * from views where
(year(payment_time)=2017 and month(payment_time)=7  and day(payment_time) = 22) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 1) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 20) or
(year(payment_time)=2017 and month(payment_time)=9  and day(payment_time) = 1);


CREATE TABLE IF NOT EXISTS ctr.clicks_filtered (
	click_time				DATETIME,
    payment_time			DATETIME,
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,	
    store_id				varchar(10),
	ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (user_id, payment_time)
);

INSERT INTO clicks_filtered(click_time, payment_time, user_id, store_id, ad_id) 
select * from clicks where
(year(payment_time)=2017 and month(payment_time)=7  and day(payment_time) = 22) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 1) or
(year(payment_time)=2017 and month(payment_time)=8  and day(payment_time) = 20) or
(year(payment_time)=2017 and month(payment_time)=9  and day(payment_time) = 1);



select max(payment_time), min(payment_time) from transactions;

select datediff(max(payment_time), min(payment_time)) from transactions;

-- There are 44 days of data in the full dataset, 4 days in the filtered dataset




CREATE TABLE IF NOT EXISTS ctrdata (
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,
	payment_time			DATETIME,
    money					int,
    kind_pay            	varchar(5),
    kind_card				varchar(7),
    store_id				varchar(10),
    network					varchar(8),
    industry				int,
    gender					varchar(7),
    address					nvarchar(100),
    view_time				DATETIME,
    click_time				DATETIME,
    ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci,
	row_id					int,
	ad_loc					int,
    ad_label				varchar(5),
    begin_time				DATETIME,
	end_time				DATETIME,
    pic_url					varchar(200),
    ad_url					varchar(300),
    ad_desc_url				varchar(300),
    ad_Copy					NVARCHAR(100),
    min_money				varchar(5), 
    mid						varchar(5000),
    order_num				varchar(20),
    maid					TEXT(31000),
    city_id					varchar(400),
    idu_category			varchar(200),
    click_hide				varchar(5),
    price					varchar(5),
    sys						varchar(5),
    ad_network				varchar(10),
    user_gender				varchar(20),
    payment_kind			varchar(10),
    PRIMARY KEY (user_id, payment_time)
);


INSERT INTO ctrdata(user_id, payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, address, view_time, click_time, ad_id,
	row_id, ad_loc, ad_label, begin_time, end_time, pic_url, ad_url, ad_desc_url, ad_Copy, min_money, mid, order_num, maid, city_id, idu_category,
    click_hide, price, sys, ad_network, user_gender, payment_kind) 
select t.user_id, t.payment_time, money, kind_pay, kind_card, t.store_id, t.network, industry, gender, address, view_time, click_time, v.ad_id,
	row_id, ad_loc, ad_label, begin_time, end_time, pic_url, ad_url, ad_desc_url, ad_Copy, min_money, mid, order_num, maid, city_id,
	idu_category, click_hide, price, sys, ad_info.network, user_gender, payment_kind
 from transactions_filtered as t 
 left join views_filtered as v
 on t.user_id=v.user_id and t.payment_time=v.payment_time
 left join clicks_filtered as c
 on t.user_id=c.user_id and t.payment_time=c.payment_time
 left join ad_info
 on v.ad_id=ad_info.ad_id;


select * from ctrdata limit 1000;

select count(payment_time) from ctrdata;

-- There are 4113234 transactions in the final table

select count(view_time) from ctrdata;

-- There 1904361 views in the final table

select count(click_time) from ctrdata;

-- There are 131544 clicks in the final table


select min(payment_time), max(payment_time) from ctrdata;

-- start date of the dataset is 2017-07-22 00:00:00
-- end date of the dataset is 2017-09-01 14:36:35



-- Create target variable Clicked.

ALTER TABLE ctrdata
ADD clicked int;

SET SQL_SAFE_UPDATES = 0;
UPDATE ctrdata
SET clicked = CASE WHEN click_time is NULL THEN 0
					Else 1 
                    END;
SET SQL_SAFE_UPDATES = 1;

select * from ctrdata where user_id="0009g";


describe transactions;




CREATE TABLE IF NOT EXISTS ctrdata2 (
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,
	payment_time			DATETIME,
    money					int,
    kind_pay            	varchar(5),
    kind_card				varchar(7),
    store_id				varchar(10),
    network					varchar(8),
    industry				int,
    gender					varchar(7),
    address					nvarchar(100),
    view_time				DATETIME,
    click_time				DATETIME,
    ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci,
	row_id					int,
	ad_loc					int,
    ad_label				varchar(5),
    begin_time				DATETIME,
	end_time				DATETIME,
    pic_url					varchar(200),
    ad_url					varchar(300),
    ad_desc_url				varchar(300),
    ad_Copy					NVARCHAR(100),
    min_money				varchar(5), 
    mid						varchar(5000),
    order_num				varchar(20),
    maid					TEXT(31000),
    city_id					varchar(400),
    idu_category			varchar(200),
    click_hide				varchar(5),
    price					varchar(5),
    sys						varchar(5),
    ad_network				varchar(10),
    user_gender				varchar(20),
    payment_kind			varchar(10),
    PRIMARY KEY (user_id, payment_time)
);





INSERT INTO ctrdata2(user_id, payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, address, view_time, click_time, ad_id,
	row_id, ad_loc, ad_label, begin_time, end_time, pic_url, ad_url, ad_desc_url, ad_Copy, min_money, mid, order_num, maid, city_id, idu_category,
    click_hide, price, sys, ad_network, user_gender, payment_kind) 
select v.user_id, v.payment_time, money, kind_pay, kind_card, v.store_id, t.network, industry, gender, address, view_time, click_time, v.ad_id,
	row_id, ad_loc, ad_label, begin_time, end_time, pic_url, ad_url, ad_desc_url, ad_Copy, min_money, mid, order_num, maid, city_id,
	idu_category, click_hide, price, sys, ad_info.network, user_gender, payment_kind
 from views_filtered as v 
 left join transactions_filtered as t
 on t.user_id=v.user_id and t.payment_time=v.payment_time
 left join clicks_filtered as c
 on t.user_id=c.user_id and t.payment_time=c.payment_time
 left join ad_info
 on v.ad_id=ad_info.ad_id;

select * from ctrdata limit 1000;

use ctr;
select count(*) from ad_info where ad_url is null;

select count(distinct ad_id) from ad_info;
