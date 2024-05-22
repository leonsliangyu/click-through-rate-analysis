
create database if not exists ctr;
use ctr;


CREATE TABLE IF NOT EXISTS ctr.transactions (
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,
	payment_time			DATETIME,
    money					int,
    kind_pay            	varchar(5),
    kind_card				varchar(7),
    store_id				varchar(10),
    network					varchar(8),
    industry				int,
    gender					varchar(7),
    address					NVARCHAR(100),
    PRIMARY KEY (user_id, payment_time)
);
-- truncate ctr.transactions;


-- @dummy used because some trans.csv's adress string contain delimiters ","

load data local infile "D:/MidTerm/trans_4.csv" 
into table ctr.transactions
fields terminated by ','
lines terminated by '\n'
(user_id, @payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, @address, @dummy1, @dummy2, @dummy3, @dummy4)
SET payment_time  = STR_TO_DATE(@payment_time, '%Y-%m-%d %H:%i:%s'),
	address= CASE
			WHEN @dummy4 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3, ' ', @dummy4)
            WHEN @dummy3 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3)
            WHEN @dummy2 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2)
            WHEN @dummy1 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1)
			ELSE @address 
	END;

load data local infile "D:/MidTerm/trans_5.csv" 
into table ctr.transactions
fields terminated by ','
lines terminated by '\n'
(user_id, @payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, @address, @dummy1, @dummy2, @dummy3, @dummy4)
SET payment_time  = STR_TO_DATE(@payment_time, '%Y-%m-%d %H:%i:%s'),
	address= CASE
			WHEN @dummy4 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3, ' ', @dummy4)
            WHEN @dummy3 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3)
            WHEN @dummy2 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2)
            WHEN @dummy1 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1)
			ELSE @address 
	END;

load data local infile "D:/MidTerm/trans_6.csv" 
into table ctr.transactions
fields terminated by ','
lines terminated by '\n'
(user_id, @payment_time, money, kind_pay, kind_card, store_id, network, industry, gender, @address, @dummy1, @dummy2, @dummy3, @dummy4)
SET payment_time  = STR_TO_DATE(@payment_time, '%Y-%m-%d %H:%i:%s'),
	address= CASE
			WHEN @dummy4 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3, ' ', @dummy4)
            WHEN @dummy3 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2, ' ', @dummy3)
            WHEN @dummy2 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1, ' ', @dummy2)
            WHEN @dummy1 IS NOT NULL THEN CONCAT(@address, ' ', @dummy1)
			ELSE @address 
	END;

select * from transactions limit 10;
select * from transactions where user_id='vV0ZjZ';



CREATE TABLE IF NOT EXISTS ctr.views (
	view_time				DATETIME,
    payment_time			DATETIME,
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,		
    store_id				varchar(10),
	ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (user_id, payment_time)
);
-- truncate ctr.views;

load data local infile "D:/MidTerm/aug-view-01-09.csv" 
into table ctr.views
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@view_time, @payment_time, user_id, store_id, ad_id)
SET view_time = STR_TO_DATE(REPLACE(@view_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');


load data local infile "D:/MidTerm/aug-view-10.csv" 
into table ctr.views
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@view_time, @payment_time, user_id, store_id, ad_id)
SET view_time = STR_TO_DATE(REPLACE(@view_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');


load data local infile "D:/MidTerm/aug-view-11-31.csv" 
into table ctr.views
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@view_time, @payment_time, user_id, store_id, ad_id)
SET view_time = STR_TO_DATE(REPLACE(@view_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');


select * from views limit 10;
select * from views where user_id='DreXYl';



CREATE TABLE IF NOT EXISTS ctr.clicks (
	click_time				DATETIME,
    payment_time			DATETIME,
	user_id					varchar(12) COLLATE utf8mb4_0900_ai_ci NOT NULL,	
    store_id				varchar(10),
	ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (user_id, payment_time)
);
-- truncate ctr.clicks;

load data local infile "D:/MidTerm/aug-click-01-09.csv" 
into table ctr.clicks
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@click_time, @payment_time, user_id, store_id, ad_id)
SET click_time = STR_TO_DATE(REPLACE(@click_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');

load data local infile "D:/MidTerm/aug-click-10.csv" 
into table ctr.clicks
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@click_time, @payment_time, user_id, store_id, ad_id)
SET click_time = STR_TO_DATE(REPLACE(@click_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');
    
load data local infile "D:/MidTerm/aug-click-11-31.csv" 
into table ctr.clicks
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(@click_time, @payment_time, user_id, store_id, ad_id)
SET click_time = STR_TO_DATE(REPLACE(@click_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    payment_time  = STR_TO_DATE(REPLACE(@payment_time, '"', ''), '%Y-%m-%d %H:%i:%s');


select * from clicks limit 10;
select * from clicks where user_id='GrX9wr';
    
    

CREATE TABLE IF NOT EXISTS ctr.ad_info (
	row_id					int,
    ad_id					varchar(10) COLLATE utf8mb4_0900_ai_ci NOT NULL,
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
    network					varchar(10),
    user_gender				varchar(20),
    payment_kind			varchar(10),
    PRIMARY KEY (ad_id)
);
-- truncate ctr.ad_info;

load data local infile "D:/MidTerm/aug-ad-info-with-tags.csv" 
into table ctr.ad_info
fields terminated by ',' ENCLOSED BY '"'
lines terminated by '\n'
(row_id,ad_id,ad_loc,ad_label,@begin_time,@end_time,pic_url,ad_url,ad_desc_url,ad_Copy,min_money, 
mid,order_num,maid,city_id,idu_category,click_hide,price,sys,network,user_gender,payment_kind)
SET begin_time = STR_TO_DATE(REPLACE(@begin_time, '"', ''), '%Y-%m-%d %H:%i:%s'),
    end_time  = STR_TO_DATE(REPLACE(@end_time, '"', ''), '%Y-%m-%d %H:%i:%s');


select * from ad_info limit 10;
select * from ad_info where ad_id='4dlZ';